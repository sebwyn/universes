mod renderer;

use vulkano::{
    pipeline::graphics::viewport::Viewport,
    buffer::{BufferUsage, CpuAccessibleBuffer},
    sync::{FenceSignalFuture, GpuFuture},
    swapchain::{self, AcquireError},
    sync::{self, FlushError},
};

use winit::{
    event::{Event, WindowEvent},
    event_loop::ControlFlow,
};

use bytemuck::{Zeroable, Pod};

use std::sync::Arc;

#[repr(C)]
#[derive(Default, Copy, Clone, Zeroable, Pod)]
struct Vertex {
    position: [f32; 2],
}

//load our shaders at compile time
mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "assets/shaders/triangle.vs"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "assets/shaders/triangle.fs"
    }
}

fn main() {
    
    //vulkan initialization
    let mut state = renderer::init_vulkan();
    let render_pass = renderer::get_render_pass(state.device.clone(), &state.image_format);
    let mut framebuffers = renderer::get_frame_buffers(&state.swapchain_images, render_pass.clone());

    vulkano::impl_vertex!(Vertex, position);

    let vertex1 = Vertex {
        position: [-0.5, -0.5],
    };
    let vertex2 = Vertex {
        position: [0.0, 0.5],
    };
    let vertex3 = Vertex {
        position: [0.5, -0.25],
    };

    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        state.device.clone(),
        BufferUsage::vertex_buffer(),
        false,
        vec![vertex1, vertex2, vertex3].into_iter(),
    )
    .unwrap();
    
    let vs = vs::load(state.device.clone()).expect("Failed to load vertex shader");
    let fs = fs::load(state.device.clone()).expect("Failed to load fragment shader");

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: state.surface.window().inner_size().into(),
        depth_range: 0.0..1.0,
    }; 

    let pipeline = renderer::get_pipeline::<Vertex>(
        state.device.clone(),
        vs.clone(),
        fs.clone(),
        render_pass.clone(),
        viewport.clone()
    );
    
    let mut command_buffers = renderer::get_command_buffers(
        state.device.clone(),
        state.queue.clone(),
        pipeline,
        &framebuffers,
        vertex_buffer.clone()
    );

    let mut window_resized = false;

    let frames_in_flight = state.swapchain_images.len();
    let mut fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; frames_in_flight];
    let mut previous_fence_i = 0;

    state.event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll; 

        match event {
            Event::WindowEvent { 
                event: WindowEvent::CloseRequested, 
                window_id: id 
            } if id == state.surface.window().id() => {
                *control_flow = ControlFlow::Exit; 
            },
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                window_resized = true;
            },
            Event::RedrawEventsCleared => {
            }
            _  => {}
        }

        if window_resized {
            window_resized = false;

            if let Some(values) = 
                renderer::recreate_swapchain(state.surface.clone(), state.swapchain.clone(), render_pass.clone()) 
            {
                (state.swapchain, state.swapchain_images, framebuffers) = values;
            };

            viewport.dimensions = state.surface.window().inner_size().into();
            let new_pipeline = renderer::get_pipeline::<Vertex>(
                state.device.clone(),
                vs.clone(),
                fs.clone(),
                render_pass.clone(),
                viewport.clone()
            );
            command_buffers = renderer::get_command_buffers(
                state.device.clone(),
                state.queue.clone(),
                new_pipeline,
                &framebuffers,
                vertex_buffer.clone()
            );
        };


        //update the game here
        let (image_i, suboptimal, acquire_future) =
            match swapchain::acquire_next_image(state.swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    window_resized = true;
                    return;
                },
                Err(_e) => return,//panic!("Failed to acquire next image {:?}", e),
            };

        if suboptimal {
            window_resized = true;
        }

        if let Some(image_fence) = &fences[image_i] {
            image_fence.wait(None).unwrap()
        }

        let previous_future = match fences[previous_fence_i].clone() {
            None => {
                let mut now = sync::now(state.device.clone());
                now.cleanup_finished();

                now.boxed()
            }
            
            Some(fence) => fence.boxed()
        };


        //if we fail to execute a command buffer, gracefully set this fence to none
        let result = previous_future
            .join(acquire_future)
            .then_execute(state.queue.clone(), command_buffers[image_i].clone());
        
        if let Ok(cb_future) = result {
            let future = cb_future
                .then_swapchain_present(state.queue.clone(), state.swapchain.clone(), image_i)
                .then_signal_fence_and_flush();

            fences[image_i] = match future {
                Ok(value) => Some(Arc::new(value)),
                Err(FlushError::OutOfDate) => {
                    window_resized = true;
                    None
                },
                Err(e) => {
                    println!("Failed to flush future {:?}", e);
                    None
                }
            };
        } else {
            window_resized = true;
            return
        }

        previous_fence_i = image_i;
    });
}
