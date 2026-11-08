#![allow(dead_code)]
use std::io::Write;

pub enum V<'a> {
    U32(u32),
    Bool(bool),
    Str(&'a str),
}

pub fn write_gguf(path: &std::path::Path, kv: &[(&str, V)], tensors: &[(&str, &[u64], u32)]) {
    let mut f = std::fs::File::create(path).unwrap();
    let s = |f: &mut std::fs::File, t: &str| {
        f.write_all(&(t.len() as u64).to_le_bytes()).unwrap();
        f.write_all(t.as_bytes()).unwrap();
    };
    f.write_all(b"GGUF").unwrap();
    f.write_all(&3u32.to_le_bytes()).unwrap();
    f.write_all(&(tensors.len() as u64).to_le_bytes()).unwrap();
    f.write_all(&(kv.len() as u64).to_le_bytes()).unwrap();
    for (key, value) in kv {
        s(&mut f, key);
        match value {
            V::U32(n) => {
                f.write_all(&4u32.to_le_bytes()).unwrap();
                f.write_all(&n.to_le_bytes()).unwrap();
            }
            V::Bool(b) => {
                f.write_all(&7u32.to_le_bytes()).unwrap();
                f.write_all(&[u8::from(*b)]).unwrap();
            }
            V::Str(t) => {
                f.write_all(&8u32.to_le_bytes()).unwrap();
                s(&mut f, t);
            }
        }
    }
    let mut offset = 0u64;
    for (name, dims, ty) in tensors {
        s(&mut f, name);
        f.write_all(&(dims.len() as u32).to_le_bytes()).unwrap();
        for d in *dims {
            f.write_all(&d.to_le_bytes()).unwrap();
        }
        f.write_all(&ty.to_le_bytes()).unwrap();
        f.write_all(&offset.to_le_bytes()).unwrap();
        offset += dims.iter().product::<u64>() * 4;
    }
}
