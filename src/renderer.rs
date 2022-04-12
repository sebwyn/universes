use vulkano::{
    instance::{Instance, InstanceCreateInfo},
    device::{
        physical::{QueueFamily, PhysicalDevice, PhysicalDeviceType},
        Device,
        DeviceCreateInfo,
        QueueCreateInfo,
        DeviceExtensions,
        Queue,
    },
    format::Format,
    render_pass::{RenderPass, Subpass, Framebuffer, FramebufferCreateInfo},
    image::{
        ImageUsage,
        SwapchainImage,
        view::ImageView
    }, 
    swapchain::{self, Surface, Swapchain, SwapchainCreateInfo, AcquireError, SwapchainCreationError},
    buffer::{BufferUsage, CpuAccessibleBuffer, BufferAccess},
    pipeline::{
        GraphicsPipeline,
        graphics::{
           input_assembly::InputAssemblyState,  
           vertex_input::{BuffersDefinition, VertexBuffersCollection},
           viewport::{Viewport, ViewportState},
        }
    },
    shader::ShaderModule,
    command_buffer::{PrimaryAutoCommandBuffer, AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents},
    sync::{self, GpuFuture, FlushError, FenceSignalFuture},
};

use vulkano_win::VkSurfaceBuild;


use winit::{
    event_loop::EventLoop,
    window::{WindowBuilder, Window}
};

use std::sync::Arc;

pub struct VulkanState<'a> {
    pub instance: Arc<Instance>,
    pub event_loop: EventLoop<()>,
    pub surface: Arc<Surface<Window>>,
    pub physical_device: Option<PhysicalDevice<'a>>,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub swapchain: Arc<Swapchain<Window>>,
    pub swapchain_images: Vec<Arc<SwapchainImage<Window>>>,
    pub image_format: Format,
}

pub fn init_vulkan<'a>() -> VulkanState<'a> {
    let extensions = vulkano_win::required_extensions();
    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };

    let instance = Instance::new(InstanceCreateInfo {
        enabled_extensions: extensions,
        ..Default::default()
    }).unwrap();

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let (physical_device, queue_family) = select_physical_device(&instance, surface.clone(), &device_extensions);

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
            enabled_extensions: physical_device
                .required_extensions()
                .union(&device_extensions),
            ..Default::default()
        }
    ).unwrap();

    let queue = queues.next().unwrap();

    let caps = physical_device
        .surface_capabilities(&surface, Default::default())
        .expect("Failed to get surface capabilities");
    let dimensions = surface.window().inner_size();
    let composite_alpha = caps.supported_composite_alpha.iter().next().unwrap();
    let image_format = Some(
        physical_device   
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0,
    );

    let (swapchain, images) = Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: caps.min_image_count + 1,
            image_format,
            image_extent: dimensions.into(),
            image_usage: ImageUsage::color_attachment(),
            composite_alpha,
            ..Default::default()
        }
    ).unwrap();
    
    VulkanState {
        instance,
        event_loop,
        surface,
        physical_device: None,
        device,
        queue,
        swapchain,
        swapchain_images: images,
        image_format: image_format.unwrap(),
    }

}

fn select_physical_device<'b>(
    instance: &'b Arc<Instance>,
    _surface: Arc<Surface<Window>>,
    device_extensions: &DeviceExtensions,
) -> (PhysicalDevice<'b>, QueueFamily<'b>) {
    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
        .filter_map(|p| {
            p.queue_families()
                .find(|&q| q.supports_graphics()) //removed a surface.is_supported because this no longer exists
                .map(|q| (p, q))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
        })
        .expect("no device available");

    (physical_device, queue_family)
}

pub fn get_render_pass(device: Arc<Device>, image_format: &Format) -> Arc<RenderPass> {
    vulkano::single_pass_renderpass!(device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: *image_format,
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    ).unwrap()
}

pub fn get_frame_buffers(images: &[Arc<SwapchainImage<Window>>], render_pass: Arc<RenderPass>) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(), 
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                }
            ).unwrap()
        })
        .collect::<Vec<_>>()
}

pub fn get_pipeline<T>(
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport
) -> Arc<GraphicsPipeline> 
where
    T: vulkano::pipeline::graphics::vertex_input::Vertex
{
    GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<T>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap()
}

pub fn get_command_buffers<T>(
    device: Arc<Device>,
    queue: Arc<Queue>,
    pipeline: Arc<GraphicsPipeline>,
    framebuffers: &Vec<Arc<Framebuffer>>,
    vertex_buffer: Arc<CpuAccessibleBuffer<[T]>>
) -> Vec<Arc<PrimaryAutoCommandBuffer>> 
where
    T: std::marker::Sync + std::marker::Send + bytemuck::Pod
{
    framebuffers
        .iter()
        .map(|framebuffer| {
            let mut builder = AutoCommandBufferBuilder::primary(
                device.clone(),
                queue.family(),
                CommandBufferUsage::MultipleSubmit
            ).unwrap();

            builder
                .begin_render_pass(
                    framebuffer.clone(),
                    SubpassContents::Inline,
                    vec![[0.0, 0.0, 0.0, 1.0].into()]
                )
                .unwrap()
                .bind_pipeline_graphics(pipeline.clone())
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .draw(vertex_buffer.read().unwrap().len() as u32, 1, 0, 0)
                .unwrap()
                .end_render_pass()
                .unwrap();

            Arc::new(builder.build().unwrap())
        })
        .collect()
}

pub fn recreate_swapchain(
    surface: Arc<Surface<Window>>,
    swapchain: Arc<Swapchain<Window>>,    
    render_pass: Arc<RenderPass>
) -> Option<(Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>, Vec<Arc<Framebuffer>>)> {

    let new_dimensions = surface.window().inner_size();

    let (new_swapchain, new_images) = match swapchain.recreate(SwapchainCreateInfo {
        image_extent: new_dimensions.into(),
        ..swapchain.create_info()
    }) {
        Ok(r) => r,
        Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return None,
        Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
    }; 
    let new_framebuffers = get_frame_buffers(&new_images, render_pass.clone());

    Some((new_swapchain, new_images, new_framebuffers))
}
