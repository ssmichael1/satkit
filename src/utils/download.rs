use anyhow::Result;
use std::path::Path;

pub fn download_if_not_exist(fname: &Path, seturl: Option<&str>) -> Result<()> {
    if fname.is_file() {
        return Ok(());
    }
    let baseurl = seturl.unwrap_or("https://storage.googleapis.com/astrokit-astro-data/");
    let url = format!(
        "{}{}",
        baseurl,
        fname.file_name().unwrap().to_str().unwrap()
    );
    // Try to set proxy, if any, from environment variables
    let agent = ureq::Agent::new_with_defaults();

    let mut resp = agent.get(url.as_str()).call()?;

    let mut dest = std::fs::File::create(fname)?;
    std::io::copy(&mut resp.body_mut().as_reader(), &mut dest)?;
    Ok(())
}

pub fn download_file(url: &str, downloaddir: &Path, overwrite_if_exists: bool) -> Result<bool> {
    let fname = std::path::Path::new(url).file_name().unwrap();
    let fullpath = downloaddir.join(fname);
    if fullpath.exists() && !overwrite_if_exists {
        println!("File {} exists; skipping download", fname.to_str().unwrap());
        Ok(false)
    } else {
        println!("Downloading {}", fname.to_str().unwrap());

        // Try to set proxy, if any, from environment variables
        let agent = ureq::Agent::new_with_defaults();

        let mut resp = agent.get(url).call()?;

        let mut dest = std::fs::File::create(fullpath)?;
        std::io::copy(&mut resp.body_mut().as_reader(), &mut dest)?;
        Ok(true)
    }
}

pub fn download_file_async(
    url: String,
    downloaddir: &Path,
    overwrite_if_exists: bool,
) -> std::thread::JoinHandle<Result<bool>> {
    let dclone = downloaddir.to_path_buf();
    let urlclone = url;
    let overwriteclone = overwrite_if_exists;
    std::thread::spawn(move || download_file(urlclone.as_str(), &dclone, overwriteclone))
}

pub fn download_to_string(url: &str) -> Result<String> {
    let agent = ureq::Agent::new_with_defaults();
    let mut resp = agent.get(url).call()?;
    let thestring = std::io::read_to_string(resp.body_mut().as_reader())?;
    Ok(thestring)
}
