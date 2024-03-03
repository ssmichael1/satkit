use crate::SKResult;
use std::path::PathBuf;

pub fn download_if_not_exist(fname: &PathBuf, seturl: Option<&str>) -> SKResult<()> {
    if fname.is_file() {
        return Ok(());
    }
    let baseurl = match seturl {
        Some(v) => v,
        None => "https://storage.googleapis.com/astrokit-astro-data/",
    };
    let url = format!(
        "{}{}",
        baseurl,
        fname.file_name().unwrap().to_str().unwrap()
    );
    // Try to set proxy, if any, from environment variables
    let agent = ureq::AgentBuilder::new().try_proxy_from_env(true).build();

    let resp = agent.get(url.as_str()).call()?;

    let mut dest = std::fs::File::create(fname)?;
    std::io::copy(resp.into_reader().as_mut(), &mut dest)?;
    Ok(())
}

pub fn download_file(
    url: &str,
    downloaddir: &PathBuf,
    overwrite_if_exists: bool,
) -> SKResult<bool> {
    let fname = std::path::Path::new(url).file_name().unwrap();
    let fullpath = downloaddir.join(fname);
    if fullpath.exists() && !overwrite_if_exists {
        println!("File {} exists; skipping download", fname.to_str().unwrap());
        Ok(false)
    } else {
        println!("Downloading {}", fname.to_str().unwrap());

        // Try to set proxy, if any, from environment variables
        let agent = ureq::AgentBuilder::new().try_proxy_from_env(true).build();

        let resp = agent.get(url).call()?;

        let mut dest = std::fs::File::create(fullpath)?;
        std::io::copy(resp.into_reader().as_mut(), &mut dest)?;
        Ok(true)
    }
}

pub fn download_file_async(
    url: String,
    downloaddir: &PathBuf,
    overwrite_if_exists: bool,
) -> std::thread::JoinHandle<SKResult<bool>> {
    let dclone = downloaddir.clone();
    let urlclone = url.clone();
    let overwriteclone = overwrite_if_exists.clone();
    std::thread::spawn(move || download_file(urlclone.as_str(), &dclone, overwriteclone))
}

pub fn download_to_string(url: &str) -> SKResult<String> {
    let agent = ureq::AgentBuilder::new().try_proxy_from_env(true).build();
    let resp = agent.get(url).call()?;
    let thestring = std::io::read_to_string(resp.into_reader().as_mut())?;
    Ok(thestring)
}
