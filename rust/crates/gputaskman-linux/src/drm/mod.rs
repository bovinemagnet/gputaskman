//! Linux AMD GPU backend: cards and device metrics from sysfs, per-process use from DRM fdinfo.
pub mod cards;
pub mod device;
pub mod fdinfo;
