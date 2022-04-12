use std::fs;
use std::env;
use std::path::{Path, PathBuf};

fn main() {
    //move assets into the target directory
    let out_dir = env::var_os("OUT_DIR").unwrap().into_string().unwrap();

    copy("assets", &format!("{}/{}", out_dir, "/assets")).expect("Failed to copy");
}

fn copy<U: AsRef<Path>>(from: U, to: U) -> std::io::Result<()> {
    let mut stack = Vec::new();
    stack.push(PathBuf::from(from.as_ref()));

    let output_root = PathBuf::from(to.as_ref());
    let input_root = PathBuf::from(from.as_ref()).components().count();

    while let Some(working_path) = stack.pop() {
        println!("process: {:?}", &working_path);

        let src: PathBuf = working_path.components().skip(input_root).collect();
        
        let dest = if src.components().count() == 0 {
            output_root.clone()
        } else {
            output_root.join(&src)
        };
        if fs::metadata(&dest).is_err() {
            fs::create_dir_all(&dest)?
        }

        for entry in fs::read_dir(working_path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                match path.file_name() {
                    Some(filename) => {
                        let dest_path = dest.join(filename);
                        fs::copy(&path, &dest_path)?;
                    },
                    None => {
                        println!("Failed {:?}", path);
                    }
                }
            }
        }
    }

    Ok(())
}
