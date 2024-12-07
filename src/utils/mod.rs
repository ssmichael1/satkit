pub use skerror::SKErr;
pub use skerror::SKResult;

mod datadir;
mod skerror;
pub use datadir::data_found;
pub use datadir::datadir;
pub use datadir::set_datadir;
pub(crate) use skerror::skerror;

#[cfg(test)]
pub mod test;

mod update_data;
pub use update_data::update_datafiles;

mod download;
pub use download::download_file;
pub use download::download_file_async;
pub use download::download_if_not_exist;
pub use download::download_to_string;

///
/// Return git hash of compiled library
///
pub fn githash<'a>() -> &'a str {
    env!("GIT_HASH")
}

///
/// Return git tag of compiled library
///
pub fn gittag<'a>() -> &'a str {
    env!("GIT_TAG")
}

///
/// Return libary compile date
///
pub fn build_date<'a>() -> &'a str {
    env!("BUILD_DATE")
}
