use std::path::Path;
use std::path::PathBuf;

pub fn get_project_root() -> std::io::Result<PathBuf> {
    let path = std::env::current_dir()?;
    let path_ancestors = path.as_path().ancestors();

    for p in path_ancestors {
        let has_cargo = std::fs::read_dir(p)?
            .any(|p| p.unwrap().file_name() == *"Cargo.lock");
        if has_cargo {
            return Ok(PathBuf::from(p));
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "Ran out of places to find Cargo.toml",
    ))
}

pub fn get_testvec_dir() -> std::io::Result<PathBuf> {
    if let Ok(val) = std::env::var("SATKIT_TESTVEC_ROOT") {
        return Ok(Path::new(&val).to_path_buf());
    }
    let root = get_project_root()?;
    Ok(root.join("satkit-testvecs"))
}
