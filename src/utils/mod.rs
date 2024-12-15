mod datadir;
pub use datadir::data_found;
pub use datadir::datadir;
pub use datadir::set_datadir;

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
