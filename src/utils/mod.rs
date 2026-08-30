pub mod datadir;
pub use datadir::add_search_dir;
pub use datadir::data_found;
pub use datadir::datadir;
pub use datadir::find_file as find_data_file;
pub use datadir::search_dirs as data_search_dirs;
pub use datadir::set_datadir;

#[cfg(test)]
pub mod test;

#[cfg(feature = "download")]
pub mod update_data;
#[cfg(feature = "download")]
pub use update_data::update_datafiles;

pub mod singleton;
pub use singleton::RefreshableSingleton;

pub mod download;
pub mod embedded;
pub mod manifest;
pub use download::download_file;
pub use download::download_file_async;
pub use download::download_if_not_exist;
pub use download::download_to_string;
pub use download::{is_offline, set_offline, OFFLINE_ENV};
pub use manifest::{fetch_static_file, Manifest, ManifestEntry};

///
/// Return git hash of compiled library
///
pub const fn githash<'a>() -> &'a str {
    env!("GIT_HASH")
}

///
/// Return git tag of compiled library
///
pub const fn gittag<'a>() -> &'a str {
    env!("GIT_TAG")
}

///
/// Return libary compile date
///
pub const fn build_date<'a>() -> &'a str {
    env!("BUILD_DATE")
}
