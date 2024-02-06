mod utils;
pub mod error;
pub use error::Error;

use byteorder::{ReadBytesExt, LE};
use error::ParseError;
use std::{io::{Cursor, Read, Seek}, u8};

use wasm_bindgen::prelude::*;
use js_sys::{Array, Function, JsString, Map, Object, Uint8Array};

use serde::{Deserialize, Serialize};

type TResult<T> = Result<T, Error>;

// When the wee_alloc feature is enabled, use wee_alloc as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

fn map_to_object(map: &Map) -> JsValue {
    match Object::from_entries(map) {
        Ok(obj) => obj.into(),
        Err(e) => e
    }
}

trait Readable<R: Read + Seek> {
    fn read(reader: &mut Context<R>) -> TResult<Self>
    where
        Self: Sized;
}

struct SeekReader<R: Read> {
    reader: R,
    read_bytes: usize,
}

fn report_progress<R: Read + Seek>(reader: &mut R, progress: &Option<Function>) {
    let offset = match reader.stream_position() {
        Ok(pos) => pos,
        Err(_) => 0
    };

    match progress {
        Some(f) => {
            let _err = f.call1(&JsValue::null(), &JsValue::from(offset));
            if let Err(_) = _err {
                panic!("Unable to invoke progress callback");
            }
        },
        None => {}
    }
}

impl<R: Read> SeekReader<R> {
    fn new(reader: R) -> Self {
        Self {
            reader,
            read_bytes: 0,
        }
    }
}
impl<R: Read> Seek for SeekReader<R> {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        match pos {
            std::io::SeekFrom::Current(0) => Ok(self.read_bytes as u64),
            _ => unimplemented!(),
        }
    }
}
impl<R: Read> Read for SeekReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.reader.read(buf).map(|s| {
            self.read_bytes += s;
            s
        })
    }
}

fn read_optional_uuid<R: Read + Seek>(reader: &mut Context<R>) -> TResult<Option<String>> {
    Ok(if reader.read_u8()? > 0 {
        Some(uuid::Uuid::read(reader)?.to_string())
    } else {
        None
    })
}

fn read_string<R: Read + Seek>(reader: &mut Context<R>) -> TResult<String> {
    let len = reader.read_i32::<LE>()?;
    if len < 0 {
        let chars = read_array((-len) as u32, reader, |r| Ok(r.read_u16::<LE>()?))?;
        let length = chars.iter().position(|&c| c == 0).unwrap_or(chars.len());
        Ok(String::from_utf16(&chars[..length]).unwrap())
    } else {
        let mut chars = vec![0; len as usize];
        reader.read_exact(&mut chars)?;
        let length = chars.iter().position(|&c| c == 0).unwrap_or(chars.len());
        Ok(String::from_utf8_lossy(&chars[..length]).into_owned())
    }
}


pub type Properties = Map;
fn read_properties_until_none<R: Read + Seek>(reader: &mut Context<R>, progress: &Option<Function>) -> TResult<Properties> {
    let properties = Properties::new();
    while let Some((name, prop)) = read_property(reader, progress)? {
        report_progress(reader, progress);

        /*
        if name == "RawData"  {
            match &prop {
                Property::Array { id, value, .. } => {
                    match value {
                        ValueArray::Base(ValueVec::Byte(ByteArray::Byte(v))) => {
                            let buf = std::io::Cursor::new(v.as_slice());
                            let mut temp_buf = std::io::BufReader::new(buf);
                            let mut temp_reader = Context::<'_, '_, '_, '_, std::io::BufReader<Cursor<&[u8]>>> {
                                stream: &mut temp_buf,
                                header: reader.header,
                                types: reader.types,
                                scope: reader.scope,
                            };
                            if let Ok(inner_props) = read_properties_until_none(&mut temp_reader, progress) {
                                temp_reader.read_u32::<LE>()?;
                                let struct_id = uuid::Uuid::read(&mut temp_reader)?;
                                let replacement = Property::RawData { id: id.clone(), properties: inner_props, struct_id };
                                properties.set(&name.into(), &replacement.into());
                                continue;
                            }
                        }
                        _ => {},
                    }
                }
                _ => {}
            }
        }
        */

        properties.set(&name.into(), &prop.into());
    }
    Ok(properties)
}

/** Reads a name then parses the property */
fn read_property<R: Read + Seek>(reader: &mut Context<R>, progress: &Option<Function>) -> TResult<Option<(String, Property)>> {
    let name = read_string(reader)?;
    report_progress(reader, progress);

    log(name.as_str());

    if name == "None" {
        Ok(None)
    } else {
        reader.scope(&name, |reader| {
            let t = PropertyType::read(reader, progress)?;
            let size = reader.read_u64::<LE>()?;
            let value = Property::read(reader, t, size, progress)?;
            Ok(Some((name.clone(), value)))
        })
    }
}

fn read_array<T, F, R: Read + Seek>(length: u32, reader: &mut Context<R>, f: F) -> TResult<Vec<T>>
where
    F: Fn(&mut Context<R>) -> TResult<T>,
{
    (0..length).map(|_| f(reader)).collect()
}

#[rustfmt::skip]
impl<R: Read + Seek> Readable<R> for uuid::Uuid {
    fn read(reader: &mut Context<R>) -> TResult<uuid::Uuid> {
        let mut b = [0; 16];
        reader.read_exact(&mut b)?;
        Ok(uuid::Uuid::from_bytes([
            b[0x3], b[0x2], b[0x1], b[0x0],
            b[0x7], b[0x6], b[0x5], b[0x4],
            b[0xb], b[0xa], b[0x9], b[0x8],
            b[0xf], b[0xe], b[0xd], b[0xc],
        ]))
    }
}



/// Used to disambiguate types within a [`Property::Set`] or [`Property::Map`] during parsing.
#[derive(Debug, Default, Clone)]
pub struct Types {
    types: std::collections::HashMap<String, StructType>,
}
impl Types {
    /// Create an empty [`Types`] specification
    pub fn new() -> Self {
        let mut types = Self::default();
        macro_rules! add_type {
            ($key_name:expr) => {
                types.add(String::from($key_name), StructType::Struct(Some("Struct".to_string())));
            };
        }

        // Add specific palworld types...
        add_type!(".worldSaveData.CharacterSaveParameterMap.Key");
        add_type!(".worldSaveData.FoliageGridSaveDataMap.Key");
        add_type!(".worldSaveData.FoliageGridSaveDataMap.ModelMap.InstanceDataMap.Key");
        add_type!(".worldSaveData.MapObjectSpawnerInStageSaveData.Key");
        add_type!(".worldSaveData.ItemContainerSaveData.Key");
        add_type!(".worldSaveData.CharacterContainerSaveData.Key");
        return types;
    }
    /// Add a new type at the given path
    pub fn add(&mut self, path: String, t: StructType) {
        // TODO: Handle escaping of '.' in property names
        // probably should store keys as Vec<String>
        self.types.insert(path, t);
    }
}

#[derive(Debug)]
enum Scope<'p, 'n> {
    Root,
    Node {
        parent: &'p Scope<'p, 'p>,
        name: &'n str,
    },
}

impl<'p, 'n> Scope<'p, 'n> {
    fn path(&self) -> String {
        match self {
            Self::Root => "".into(),
            Self::Node { parent, name } => {
                format!("{}.{}", parent.path(), name)
            }
        }
    }
}

#[derive(Debug)]
struct Context<'stream, 'header, 'types, 'scope, S> {
    stream: &'stream mut S,
    header: Option<&'header Header>,
    types: &'types Types,
    scope: &'scope Scope<'scope, 'scope>,
}
impl<R: Read> Read for Context<'_, '_, '_, '_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.stream.read(buf)
    }
}
impl<S: Seek> Seek for Context<'_, '_, '_, '_, S> {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        self.stream.seek(pos)
    }
}

impl<'stream, 'header, 'types, 'scope, S> Context<'stream, 'header, 'types, 'scope, S> {
    fn run_with_types<F, T>(types: &'types Types, stream: &'stream mut S, f: F) -> T
    where
        F: FnOnce(&mut Context<'stream, '_, 'types, 'scope, S>) -> T,
    {
        f(&mut Context::<'stream, '_, 'types, 'scope> {
            stream,
            header: None,
            types,
            scope: &Scope::Root,
        })
    }
    fn scope<'name, F, T>(&mut self, name: &'name str, f: F) -> T
    where
        F: FnOnce(&mut Context<'_, '_, 'types, '_, S>) -> T,
    {
        f(&mut Context {
            stream: self.stream,
            header: self.header,
            types: self.types,
            scope: &Scope::Node {
                name,
                parent: self.scope,
            },
        })
    }
    fn header<'h, F, T>(&mut self, header: &'h Header, f: F) -> T
    where
        F: FnOnce(&mut Context<'_, '_, 'types, '_, S>) -> T,
    {
        f(&mut Context {
            stream: self.stream,
            header: Some(header),
            types: self.types,
            scope: self.scope,
        })
    }
    fn path(&self) -> String {
        self.scope.path()
    }
    fn get_type(&self) -> Option<&'types StructType> {
        self.types.types.get(&self.path())
    }
}
impl<'stream, 'header, 'types, 'scope, R: Read + Seek>
    Context<'stream, 'header, 'types, 'scope, R>
{
    fn get_type_or<'t>(&mut self, t: &'t StructType) -> TResult<&'t StructType>
    where
        'types: 't,
    {
        let _offset = self.stream.stream_position()?;
        Ok(self.get_type().unwrap_or_else(|| t))
    }
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropertyType {
    IntProperty,
    Int8Property,
    Int16Property,
    Int64Property,
    UInt8Property,
    UInt16Property,
    UInt32Property,
    UInt64Property,
    FloatProperty,
    DoubleProperty,
    BoolProperty,
    ByteProperty,
    EnumProperty,
    ArrayProperty,
    ObjectProperty,
    StrProperty,
    FieldPathProperty,
    SoftObjectProperty,
    NameProperty,
    TextProperty,
    DelegateProperty,
    MulticastDelegateProperty,
    MulticastInlineDelegateProperty,
    MulticastSparseDelegateProperty,
    SetProperty,
    MapProperty,
    StructProperty,
}

impl PropertyType {
    pub fn get_name(&self) -> &str {
        match &self {
            PropertyType::Int8Property => "Int8Property",
            PropertyType::Int16Property => "Int16Property",
            PropertyType::IntProperty => "IntProperty",
            PropertyType::Int64Property => "Int64Property",
            PropertyType::UInt8Property => "UInt8Property",
            PropertyType::UInt16Property => "UInt16Property",
            PropertyType::UInt32Property => "UInt32Property",
            PropertyType::UInt64Property => "UInt64Property",
            PropertyType::FloatProperty => "FloatProperty",
            PropertyType::DoubleProperty => "DoubleProperty",
            PropertyType::BoolProperty => "BoolProperty",
            PropertyType::ByteProperty => "ByteProperty",
            PropertyType::EnumProperty => "EnumProperty",
            PropertyType::ArrayProperty => "ArrayProperty",
            PropertyType::ObjectProperty => "ObjectProperty",
            PropertyType::StrProperty => "StrProperty",
            PropertyType::FieldPathProperty => "FieldPathProperty",
            PropertyType::SoftObjectProperty => "SoftObjectProperty",
            PropertyType::NameProperty => "NameProperty",
            PropertyType::TextProperty => "TextProperty",
            PropertyType::DelegateProperty => "DelegateProperty",
            PropertyType::MulticastDelegateProperty => "MulticastDelegateProperty",
            PropertyType::MulticastInlineDelegateProperty => "MulticastInlineDelegateProperty",
            PropertyType::MulticastSparseDelegateProperty => "MulticastSparseDelegateProperty",
            PropertyType::SetProperty => "SetProperty",
            PropertyType::MapProperty => "MapProperty",
            PropertyType::StructProperty => "StructProperty",
        }
    }
    fn read<R: Read + Seek>(reader: &mut Context<R>, progress: &Option<Function>) -> TResult<Self> {
        let t = read_string(reader)?;
        report_progress(reader, progress);

        match t.as_str() {
            "Int8Property" => Ok(PropertyType::Int8Property),
            "Int16Property" => Ok(PropertyType::Int16Property),
            "IntProperty" => Ok(PropertyType::IntProperty),
            "Int64Property" => Ok(PropertyType::Int64Property),
            "UInt8Property" => Ok(PropertyType::UInt8Property),
            "UInt16Property" => Ok(PropertyType::UInt16Property),
            "UInt32Property" => Ok(PropertyType::UInt32Property),
            "UInt64Property" => Ok(PropertyType::UInt64Property),
            "FloatProperty" => Ok(PropertyType::FloatProperty),
            "DoubleProperty" => Ok(PropertyType::DoubleProperty),
            "BoolProperty" => Ok(PropertyType::BoolProperty),
            "ByteProperty" => Ok(PropertyType::ByteProperty),
            "EnumProperty" => Ok(PropertyType::EnumProperty),
            "ArrayProperty" => Ok(PropertyType::ArrayProperty),
            "ObjectProperty" => Ok(PropertyType::ObjectProperty),
            "StrProperty" => Ok(PropertyType::StrProperty),
            "FieldPathProperty" => Ok(PropertyType::FieldPathProperty),
            "SoftObjectProperty" => Ok(PropertyType::SoftObjectProperty),
            "NameProperty" => Ok(PropertyType::NameProperty),
            "TextProperty" => Ok(PropertyType::TextProperty),
            "DelegateProperty" => Ok(PropertyType::DelegateProperty),
            "MulticastDelegateProperty" => Ok(PropertyType::MulticastDelegateProperty),
            "MulticastInlineDelegateProperty" => Ok(PropertyType::MulticastInlineDelegateProperty),
            "MulticastSparseDelegateProperty" => Ok(PropertyType::MulticastSparseDelegateProperty),
            "SetProperty" => Ok(PropertyType::SetProperty),
            "MapProperty" => Ok(PropertyType::MapProperty),
            "StructProperty" => Ok(PropertyType::StructProperty),
            _ => Err(Error::UnknownPropertyType(format!("{t:?}"))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag="_type", content="_struct")]
pub enum StructType {
    Guid,
    DateTime,
    Timespan,
    Vector2D,
    Vector,
    Box,
    IntPoint,
    Quat,
    Rotator,
    LinearColor,
    Color,
    SoftObjectPath,
    GameplayTagContainer,
    Struct(Option<String>),
}
impl From<&str> for StructType {
    fn from(t: &str) -> Self {
        match t {
            "Guid" => StructType::Guid,
            "DateTime" => StructType::DateTime,
            "Timespan" => StructType::Timespan,
            "Vector2D" => StructType::Vector2D,
            "Vector" => StructType::Vector,
            "Box" => StructType::Box,
            "IntPoint" => StructType::IntPoint,
            "Quat" => StructType::Quat,
            "Rotator" => StructType::Rotator,
            "LinearColor" => StructType::LinearColor,
            "Color" => StructType::Color,
            "SoftObjectPath" => StructType::SoftObjectPath,
            "GameplayTagContainer" => StructType::GameplayTagContainer,
            "Struct" => StructType::Struct(None),
            _ => StructType::Struct(Some(t.to_owned())),
        }
    }
}
impl From<String> for StructType {
    fn from(t: String) -> Self {
        match t.as_str() {
            "Guid" => StructType::Guid,
            "DateTime" => StructType::DateTime,
            "Timespan" => StructType::Timespan,
            "Vector2D" => StructType::Vector2D,
            "Vector" => StructType::Vector,
            "Box" => StructType::Box,
            "IntPoint" => StructType::IntPoint,
            "Quat" => StructType::Quat,
            "Rotator" => StructType::Rotator,
            "LinearColor" => StructType::LinearColor,
            "Color" => StructType::Color,
            "SoftObjectPath" => StructType::SoftObjectPath,
            "GameplayTagContainer" => StructType::GameplayTagContainer,
            "Struct" => StructType::Struct(None),
            _ => StructType::Struct(Some(t)),
        }
    }
}
impl StructType {
    fn read<R: Read + Seek>(reader: &mut Context<R>) -> TResult<Self> {
        Ok(read_string(reader)?.into())
    }
}

type DateTime = u64;
type Timespan = i64;
type Int8 = i8;
type Int16 = i16;
type Int = i32;
type Int64 = i64;
type UInt8 = u8;
type UInt16 = u16;
type UInt32 = u32;
type UInt64 = u64;
type Float = f32;
type Double = f64;
type Bool = bool;
pub type Enum = String;

#[derive(Debug, PartialEq)]
pub struct MapEntry {
    pub key: PropertyValue,
    pub value: PropertyValue,
}
impl MapEntry {
    fn read<R: Read + Seek>(
        reader: &mut Context<R>,
        key_type: &PropertyType,
        key_struct_type: Option<&StructType>,
        value_type: &PropertyType,
        value_struct_type: Option<&StructType>,
        progress: &Option<Function>
    ) -> TResult<MapEntry> {
        let key = PropertyValue::read(reader, key_type, key_struct_type, progress)?;
        let value = PropertyValue::read(reader, value_type, value_struct_type, progress)?;
        Ok(Self { key, value })
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldPath {
    path: Vec<String>,
    owner: String,
}
impl FieldPath {
    fn read<R: Read + Seek>(reader: &mut Context<R>) -> TResult<Self> {
        Ok(Self {
            path: read_array(reader.read_u32::<LE>()?, reader, read_string)?,
            owner: read_string(reader)?,
        })
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Delegate {
    pub name: String,
    pub path: String,
}
impl Delegate {
    fn read<R: Read + Seek>(reader: &mut Context<R>) -> TResult<Self> {
        Ok(Self {
            name: read_string(reader)?,
            path: read_string(reader)?,
        })
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct MulticastDelegate(Vec<Delegate>);
impl MulticastDelegate {
    fn read<R: Read + Seek>(reader: &mut Context<R>) -> TResult<Self> {
        Ok(Self(read_array(
            reader.read_u32::<LE>()?,
            reader,
            Delegate::read,
        )?))
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct MulticastInlineDelegate(Vec<Delegate>);
impl MulticastInlineDelegate {
    fn read<R: Read + Seek>(reader: &mut Context<R>) -> TResult<Self> {
        Ok(Self(read_array(
            reader.read_u32::<LE>()?,
            reader,
            Delegate::read,
        )?))
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct MulticastSparseDelegate(Vec<Delegate>);
impl MulticastSparseDelegate {
    fn read<R: Read + Seek>(reader: &mut Context<R>) -> TResult<Self> {
        Ok(Self(read_array(
            reader.read_u32::<LE>()?,
            reader,
            Delegate::read,
        )?))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinearColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl From<LinearColor> for JsValue {
    fn from(val: LinearColor) -> Self {
        let array = Array::new_with_length(4);
        array.set(0, JsValue::from(val.r));
        array.set(1, JsValue::from(val.g));
        array.set(2, JsValue::from(val.b));
        array.set(3, JsValue::from(val.a));
        array.into()
    }
}

impl LinearColor {
    fn read<R: Read + Seek>(reader: &mut Context<R>) -> TResult<Self> {
        Ok(Self {
            r: reader.read_f32::<LE>()?,
            g: reader.read_f32::<LE>()?,
            b: reader.read_f32::<LE>()?,
            a: reader.read_f32::<LE>()?,
        })
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct Quat {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl From<Quat> for JsValue {
    fn from(val: Quat) -> Self {
        let array = Array::new_with_length(4);
        array.set(0, JsValue::from(val.x));
        array.set(1, JsValue::from(val.y));
        array.set(2, JsValue::from(val.z));
        array.set(3, JsValue::from(val.w));
        array.into()
    }
}

impl Quat {
    fn read<R: Read + Seek>(reader: &mut Context<R>) -> TResult<Self> {
        if reader.header.as_ref().unwrap().large_world_coordinates() {
            Ok(Self {
                x: reader.read_f64::<LE>()?,
                y: reader.read_f64::<LE>()?,
                z: reader.read_f64::<LE>()?,
                w: reader.read_f64::<LE>()?,
            })
        } else {
            Ok(Self {
                x: reader.read_f32::<LE>()? as f64,
                y: reader.read_f32::<LE>()? as f64,
                z: reader.read_f32::<LE>()? as f64,
                w: reader.read_f32::<LE>()? as f64,
            })
        }
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct Rotator {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl From<Rotator> for JsValue {
    fn from(val: Rotator) -> Self {
        let array = Array::new_with_length(3);
        array.set(0, JsValue::from(val.x));
        array.set(1, JsValue::from(val.y));
        array.set(2, JsValue::from(val.z));
        array.into()
    }
}

impl Rotator {
    fn read<R: Read + Seek>(reader: &mut Context<R>) -> TResult<Self> {
        if reader.header.as_ref().unwrap().large_world_coordinates() {
            Ok(Self {
                x: reader.read_f64::<LE>()?,
                y: reader.read_f64::<LE>()?,
                z: reader.read_f64::<LE>()?,
            })
        } else {
            Ok(Self {
                x: reader.read_f32::<LE>()? as f64,
                y: reader.read_f32::<LE>()? as f64,
                z: reader.read_f32::<LE>()? as f64,
            })
        }
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl From<Color> for JsValue {
    fn from(val: Color) -> Self {
        let array = Array::new_with_length(4);
        array.set(0, JsValue::from(val.r));
        array.set(1, JsValue::from(val.g));
        array.set(2, JsValue::from(val.b));
        array.set(3, JsValue::from(val.a));
        array.into()
    }
}

impl Color {
    fn read<R: Read + Seek>(reader: &mut Context<R>) -> TResult<Self> {
        Ok(Self {
            r: reader.read_u8()?,
            g: reader.read_u8()?,
            b: reader.read_u8()?,
            a: reader.read_u8()?,
        })
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Vector {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl From<Vector> for JsValue {
    fn from(val: Vector) -> Self {
        let array = Array::new_with_length(3);
        array.set(0, JsValue::from(val.x));
        array.set(1, JsValue::from(val.y));
        array.set(2, JsValue::from(val.z));
        array.into()
    }
}

impl Vector {
    fn read<R: Read + Seek>(reader: &mut Context<R>) -> TResult<Self> {
        if reader.header.as_ref().unwrap().large_world_coordinates() {
            Ok(Self {
                x: reader.read_f64::<LE>()?,
                y: reader.read_f64::<LE>()?,
                z: reader.read_f64::<LE>()?,
            })
        } else {
            Ok(Self {
                x: reader.read_f32::<LE>()? as f64,
                y: reader.read_f32::<LE>()? as f64,
                z: reader.read_f32::<LE>()? as f64,
            })
        }
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct Vector2D {
    pub x: f32,
    pub y: f32,
}

impl From<Vector2D> for JsValue {
    fn from(val: Vector2D) -> Self {
        let array = Array::new_with_length(2);
        array.set(0, JsValue::from(val.x));
        array.set(1, JsValue::from(val.y));
        array.into()
    }
}

impl Vector2D {
    fn read<R: Read + Seek>(reader: &mut Context<R>) -> TResult<Self> {
        Ok(Self {
            x: reader.read_f32::<LE>()?,
            y: reader.read_f32::<LE>()?,
        })
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Box {
    pub a: Vector,
    pub b: Vector,
}

impl From<Box> for JsValue {
    fn from(value: Box) -> Self {
        let array = Array::new_with_length(2);
        array.set(0, JsValue::from(value.a));
        array.set(1, JsValue::from(value.b));
        array.into()
    }
}

impl Box {
    fn read<R: Read + Seek>(reader: &mut Context<R>) -> TResult<Self> {
        let a = Vector::read(reader)?;
        let b = Vector::read(reader)?;
        reader.read_u8()?;
        Ok(Self { a, b })
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct IntPoint {
    pub x: i32,
    pub y: i32,
}

impl From<IntPoint> for JsValue {
    fn from(val: IntPoint) -> Self {
        let array = Array::new_with_length(2);
        array.set(0, JsValue::from(val.x));
        array.set(1, JsValue::from(val.y));
        array.into()
    }
}

impl IntPoint {
    fn read<R: Read + Seek>(reader: &mut Context<R>) -> TResult<Self> {
        Ok(Self {
            x: reader.read_i32::<LE>()?,
            y: reader.read_i32::<LE>()?,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GameplayTag {
    pub name: String,
}
impl GameplayTag {
    fn read<R: Read + Seek>(reader: &mut Context<R>) -> TResult<Self> {
        Ok(Self {
            name: read_string(reader)?,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GameplayTagContainer {
    pub gameplay_tags: Vec<GameplayTag>,
}
impl GameplayTagContainer {
    fn read<R: Read + Seek>(reader: &mut Context<R>) -> TResult<Self> {
        Ok(Self {
            gameplay_tags: read_array(reader.read_u32::<LE>()?, reader, GameplayTag::read)?,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FFormatArgumentData {
    name: String,
    value: FFormatArgumentDataValue,
}
impl<R: Read + Seek> Readable<R> for FFormatArgumentData {
    fn read(reader: &mut Context<R>) -> TResult<Self> {
        Ok(Self {
            name: read_string(reader)?,
            value: FFormatArgumentDataValue::read(reader)?,
        })
    }
}
// very similar to FFormatArgumentValue but serializes ints as 32 bits (TODO changes to 64 bit
// again at some later UE version)
#[derive(Debug, Clone, PartialEq)]
pub enum FFormatArgumentDataValue {
    Int(i32),
    UInt(u32),
    Float(f32),
    Double(f64),
    Text(std::boxed::Box<Text>),
    Gender(u64),
}
impl<R: Read + Seek> Readable<R> for FFormatArgumentDataValue {
    fn read(reader: &mut Context<R>) -> TResult<Self> {
        let type_ = reader.read_u8()?;
        match type_ {
            0 => Ok(Self::Int(reader.read_i32::<LE>()?)),
            1 => Ok(Self::UInt(reader.read_u32::<LE>()?)),
            2 => Ok(Self::Float(reader.read_f32::<LE>()?)),
            3 => Ok(Self::Double(reader.read_f64::<LE>()?)),
            4 => Ok(Self::Text(std::boxed::Box::new(Text::read(reader)?))),
            5 => Ok(Self::Gender(reader.read_u64::<LE>()?)),
            _ => Err(Error::Other(format!(
                "unimplemented variant for FFormatArgumentDataValue 0x{type_:x}"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum FFormatArgumentValue {
    Int(i64),
    UInt(u64),
    Float(f32),
    Double(f64),
    Text(std::boxed::Box<Text>),
    Gender(u64),
}

impl<R: Read + Seek> Readable<R> for FFormatArgumentValue {
    fn read(reader: &mut Context<R>) -> TResult<Self> {
        let type_ = reader.read_u8()?;
        match type_ {
            0 => Ok(Self::Int(reader.read_i64::<LE>()?)),
            1 => Ok(Self::UInt(reader.read_u64::<LE>()?)),
            2 => Ok(Self::Float(reader.read_f32::<LE>()?)),
            3 => Ok(Self::Double(reader.read_f64::<LE>()?)),
            4 => Ok(Self::Text(std::boxed::Box::new(Text::read(reader)?))),
            5 => Ok(Self::Gender(reader.read_u64::<LE>()?)),
            _ => Err(Error::Other(format!(
                "unimplemented variant for FFormatArgumentValue 0x{type_:x}"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FNumberFormattingOptions {
    always_sign: bool,
    use_grouping: bool,
    rounding_mode: i8, // TODO enum ERoundingMode
    minimum_integral_digits: i32,
    maximum_integral_digits: i32,
    minimum_fractional_digits: i32,
    maximum_fractional_digits: i32,
}
impl<R: Read + Seek> Readable<R> for FNumberFormattingOptions {
    fn read(reader: &mut Context<R>) -> TResult<Self> {
        Ok(Self {
            always_sign: reader.read_u32::<LE>()? != 0,
            use_grouping: reader.read_u32::<LE>()? != 0,
            rounding_mode: reader.read_i8()?,
            minimum_integral_digits: reader.read_i32::<LE>()?,
            maximum_integral_digits: reader.read_i32::<LE>()?,
            minimum_fractional_digits: reader.read_i32::<LE>()?,
            maximum_fractional_digits: reader.read_i32::<LE>()?,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Text {
    flags: u32,
    pub variant: TextVariant,
}
#[derive(Debug, Clone, PartialEq)]
pub enum TextVariant {
    // -0x1
    None {
        culture_invariant: Option<String>,
    },
    // 0x0
    Base {
        namespace: String,
        key: String,
        source_string: String,
    },
    // 0x3
    ArgumentFormat {
        // aka ArgumentDataFormat
        format_text: std::boxed::Box<Text>,
        arguments: Vec<FFormatArgumentData>,
    },
    // 0x4
    AsNumber {
        source_value: FFormatArgumentValue,
        format_options: Option<FNumberFormattingOptions>,
        culture_name: String,
    },
    // 0x7
    AsDate {
        source_date_time: DateTime,
        date_style: i8, // TODO EDateTimeStyle::Type
        time_zone: String,
        culture_name: String,
    },
    StringTableEntry {
        // 0xb
        table: String,
        key: String,
    },
}

impl<R: Read + Seek> Readable<R> for Text {
    fn read(reader: &mut Context<R>) -> TResult<Self> {
        let flags = reader.read_u32::<LE>()?;
        let text_history_type = reader.read_i8()?;
        let variant = match text_history_type {
            -0x1 => Ok(TextVariant::None {
                culture_invariant: (reader.read_u32::<LE>()? != 0) // bHasCultureInvariantString
                    .then(|| read_string(reader))
                    .transpose()?,
            }),
            0x0 => Ok(TextVariant::Base {
                namespace: read_string(reader)?,
                key: read_string(reader)?,
                source_string: read_string(reader)?,
            }),
            0x3 => Ok(TextVariant::ArgumentFormat {
                format_text: std::boxed::Box::new(Text::read(reader)?),
                arguments: read_array(reader.read_u32::<LE>()?, reader, FFormatArgumentData::read)?,
            }),
            0x4 => Ok(TextVariant::AsNumber {
                source_value: FFormatArgumentValue::read(reader)?,
                format_options:
                    (reader.read_u32::<LE>()? != 0) // bHasFormatOptions
                        .then(|| FNumberFormattingOptions::read(reader))
                        .transpose()?,
                culture_name: read_string(reader)?,
            }),
            0x7 => Ok(TextVariant::AsDate {
                source_date_time: reader.read_u64::<LE>()?,
                date_style: reader.read_i8()?,
                time_zone: read_string(reader)?,
                culture_name: read_string(reader)?,
            }),
            0xb => Ok({
                TextVariant::StringTableEntry {
                    table: read_string(reader)?,
                    key: read_string(reader)?,
                }
            }),
            _ => Err(Error::Other(format!(
                "unimplemented variant for FTextHistory 0x{text_history_type:x}"
            ))),
        }?;
        Ok(Self { flags, variant })
    }
}

/// Just a plain byte, or an enum in which case the variant will be a String
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Byte {
    Byte(u8),
    Label(String),
}

impl From<Byte> for JsValue {
    fn from(value: Byte) -> Self {
        match value {
            Byte::Label(s) => JsValue::from(s),
            Byte::Byte(b) => JsValue::from(b)
        }
    }
}

/// Vectorized [`Byte`]
#[derive(Debug, Clone, PartialEq)]
pub enum ByteArray {
    Byte(Vec<u8>),
    Label(Vec<String>),
}

#[derive(Debug, PartialEq)]
pub enum PropertyValue {
    Int(Int),
    Int8(Int8),
    Int16(Int16),
    Int64(Int64),
    UInt16(UInt16),
    UInt32(UInt32),
    Float(Float),
    Double(Double),
    Bool(Bool),
    Byte(Byte),
    Enum(Enum),
    Name(String),
    Str(String),
    SoftObject(String, String),
    SoftObjectPath(String, String),
    Object(String),
    Struct(StructValue),
}

impl From<PropertyValue> for JsValue {
    fn from(value: PropertyValue) -> Self {
        match value {
            PropertyValue::Bool(b) => JsValue::from(b),
            PropertyValue::Int(v) => JsValue::from(v),
            PropertyValue::Int8(v) => JsValue::from(v),
            PropertyValue::Int16(v) => JsValue::from(v),
            PropertyValue::Int64(v) => JsValue::from(v),
            PropertyValue::UInt16(v) => JsValue::from(v),
            PropertyValue::UInt32(v) => JsValue::from(v),
            PropertyValue::Float(v) => JsValue::from(v),
            PropertyValue::Double(v) => JsValue::from(v),
            PropertyValue::Byte(v) => JsValue::from(v),
            PropertyValue::Enum(v) => JsValue::from(v),
            PropertyValue::Name(v) => JsValue::from(v),
            PropertyValue::Str(v) => JsValue::from(v),
            PropertyValue::Object(v) => JsValue::from(v),
            PropertyValue::Struct(v) => JsValue::from(v),
            PropertyValue::SoftObject(value, value2) |
            PropertyValue::SoftObjectPath(value, value2) => {
                let array = Array::new_with_length(2);
                array.set(0, JsValue::from(value));
                array.set(1, JsValue::from(value2));
                array.into()
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum StructValue {
    Guid(uuid::Uuid),
    DateTime(DateTime),
    Timespan(Timespan),
    Vector2D(Vector2D),
    Vector(Vector),
    Box(Box),
    IntPoint(IntPoint),
    Quat(Quat),
    LinearColor(LinearColor),
    Color(Color),
    Rotator(Rotator),
    SoftObjectPath(String, String),
    GameplayTagContainer(GameplayTagContainer),
    /// User defined struct which is simply a list of properties
    Struct(Properties),
}

impl From<StructValue> for JsValue {
    fn from(val: StructValue) -> Self {
        match val {
            StructValue::Guid(guid) => JsValue::from(guid.as_hyphenated().to_string()),
            StructValue::Box(b) => JsValue::from(b),
            StructValue::DateTime(dt) => JsValue::from(dt),
            StructValue::Timespan(i) => JsValue::from(i),
            StructValue::Vector2D(v) => JsValue::from(v),
            StructValue::Vector(v) => JsValue::from(v),
            StructValue::IntPoint(p) => JsValue::from(p),
            StructValue::Quat(q) => JsValue::from(q),
            StructValue::LinearColor(c) => JsValue::from(c),
            StructValue::Color(c) => JsValue::from(c),
            StructValue::Rotator(r) => JsValue::from(r),
            StructValue::SoftObjectPath(value, value2) => {
                let array = Array::new_with_length(2);
                array.set(0, JsValue::from(value));
                array.set(1, JsValue::from(value2));
                array.into()
            },
            StructValue::GameplayTagContainer(container) => {
                let array = Array::new();
                for tag in container.gameplay_tags {
                    array.push(&JsValue::from(tag.name));
                }
                array.into()
            },
            StructValue::Struct(properties) => map_to_object(&properties),
        }
    }
}

/// Vectorized properties to avoid storing the variant with each value
#[derive(Debug, Clone, PartialEq)]
pub enum ValueVec {
    Int8(Vec<Int8>),
    Int16(Vec<Int16>),
    Int(Vec<Int>),
    Int64(Vec<Int64>),
    UInt8(Vec<UInt8>),
    UInt16(Vec<UInt16>),
    UInt32(Vec<UInt32>),
    UInt64(Vec<UInt64>),
    Float(Vec<Float>),
    Double(Vec<Double>),
    Bool(Vec<bool>),
    Byte(ByteArray),
    Enum(Vec<Enum>),
    Str(Vec<String>),
    Text(Vec<Text>),
    SoftObject(Vec<(String, String)>),
    Name(Vec<String>),
    Object(Vec<String>),
    Box(Vec<Box>),
}

impl From<ValueVec> for JsValue {
    fn from(value: ValueVec) -> Self {
        match value {
            ValueVec::Bool(v)     => vec_to_array(v),
            ValueVec::Box(v)       => vec_to_array(v),
            ValueVec::Double(v)    => vec_to_array(v),
            ValueVec::Int8(v)       => vec_to_array(v),
            ValueVec::Int16(v)     => vec_to_array(v),
            ValueVec::Int(v)       => vec_to_array(v),
            ValueVec::Int64(v)     => vec_to_array(v),
            ValueVec::UInt8(v)      => vec_to_array(v),
            ValueVec::UInt16(v)    => vec_to_array(v),
            ValueVec::UInt32(v)    => vec_to_array(v),
            ValueVec::UInt64(v)    => vec_to_array(v),
            ValueVec::Float(v)     => vec_to_array(v),
            ValueVec::Enum(v)   => vec_to_array(v),
            ValueVec::Str(v)    => vec_to_array(v),
            ValueVec::Name(v)   => vec_to_array(v),
            ValueVec::Object(v) => vec_to_array(v),
            ValueVec::SoftObject(v) => tuple_vec_to_array(v),
            ValueVec::Byte(v) => {
                match v {
                    ByteArray::Byte(b) => vec_to_array(b),
                    ByteArray::Label(b) => vec_to_array(b)
                }
            },
            _ => Array::new()
        }.into()
    }
}

/// Encapsulates [`ValueVec`] with a special handling of structs. See also: [`ValueSet`]
#[derive(Debug, Clone, PartialEq)]
pub enum ValueArray {
    Base(ValueVec),
    Struct {
        _type: String,
        name: String,
        struct_type: StructType,
        id: String,
        value: Vec<StructValue>,
    },
}

impl From<ValueArray> for JsValue {
    fn from(value: ValueArray) -> Self {
        match value {
            ValueArray::Base(b) => JsValue::from(b),
            ValueArray::Struct { _type, name, value, ..} => {
                let map = Map::new();
                let array = Array::new();
                for sval in value {
                    array.push(&JsValue::from(sval));
                }

                map.set(&JsString::from("Name").into(), &JsValue::from(name));
                map.set(&JsString::from("Type").into(), &JsValue::from(_type));
                map.set(&JsString::from("Values").into(), &array.into());
                map_to_object(&map)
            }
        }
    }
}

/// Encapsulates [`ValueVec`] with a special handling of structs. See also: [`ValueArray`]
#[derive(Debug, PartialEq)]
pub enum ValueSet {
    Base(ValueVec),
    Struct(Vec<StructValue>),
}

impl From<ValueSet> for JsValue {
    fn from(value: ValueSet) -> Self {
        match value {
            ValueSet::Base(v) => JsValue::from(v),
            ValueSet::Struct(v) => {
                let array = Array::new();
                for _struct in v {
                    array.push(&JsValue::from(_struct));
                }
                array.into()
            }
        }
    }
}

impl PropertyValue {
    fn read<R: Read + Seek>(
        reader: &mut Context<R>,
        t: &PropertyType,
        st: Option<&StructType>,
        progress: &Option<Function>
    ) -> TResult<PropertyValue> {
        Ok(match t {
            PropertyType::IntProperty => PropertyValue::Int(reader.read_i32::<LE>()?),
            PropertyType::Int8Property => PropertyValue::Int8(reader.read_i8()?),
            PropertyType::Int16Property => PropertyValue::Int16(reader.read_i16::<LE>()?),
            PropertyType::Int64Property => PropertyValue::Int64(reader.read_i64::<LE>()?),
            PropertyType::UInt16Property => PropertyValue::UInt16(reader.read_u16::<LE>()?),
            PropertyType::UInt32Property => PropertyValue::UInt32(reader.read_u32::<LE>()?),
            PropertyType::FloatProperty => PropertyValue::Float(reader.read_f32::<LE>()?),
            PropertyType::DoubleProperty => PropertyValue::Double(reader.read_f64::<LE>()?),
            PropertyType::BoolProperty => PropertyValue::Bool(reader.read_u8()? > 0),
            PropertyType::NameProperty => PropertyValue::Name(read_string(reader)?),
            PropertyType::StrProperty => PropertyValue::Str(read_string(reader)?),
            PropertyType::SoftObjectProperty => {
                PropertyValue::SoftObject(read_string(reader)?, read_string(reader)?)
            }
            PropertyType::ObjectProperty => PropertyValue::Object(read_string(reader)?),
            PropertyType::ByteProperty => PropertyValue::Byte(Byte::Label(read_string(reader)?)),
            PropertyType::EnumProperty => PropertyValue::Enum(read_string(reader)?),
            PropertyType::StructProperty => {
                PropertyValue::Struct(StructValue::read(reader, st.as_ref().unwrap(), progress)?)
            }
            _ => return Err(Error::Other(format!("unimplemented property {t:?}"))),
        })
    }
}
impl StructValue {
    fn read<R: Read + Seek>(reader: &mut Context<R>, t: &StructType, progress: &Option<Function>) -> TResult<StructValue> {
        Ok(match t {
            StructType::Guid => StructValue::Guid(uuid::Uuid::read(reader)?),
            StructType::DateTime => StructValue::DateTime(reader.read_u64::<LE>()?),
            StructType::Timespan => StructValue::Timespan(reader.read_i64::<LE>()?),
            StructType::Vector2D => StructValue::Vector2D(Vector2D::read(reader)?),
            StructType::Vector => StructValue::Vector(Vector::read(reader)?),
            StructType::Box => StructValue::Box(Box::read(reader)?),
            StructType::IntPoint => StructValue::IntPoint(IntPoint::read(reader)?),
            StructType::Quat => StructValue::Quat(Quat::read(reader)?),
            StructType::LinearColor => StructValue::LinearColor(LinearColor::read(reader)?),
            StructType::Color => StructValue::Color(Color::read(reader)?),
            StructType::Rotator => StructValue::Rotator(Rotator::read(reader)?),
            StructType::SoftObjectPath => {
                StructValue::SoftObjectPath(read_string(reader)?, read_string(reader)?)
            },
            StructType::GameplayTagContainer => {
                StructValue::GameplayTagContainer(GameplayTagContainer::read(reader)?)
            },
            StructType::Struct(_) => {
                StructValue::Struct(read_properties_until_none(reader, progress)?)
            }
        })
    }
}
impl ValueVec {
    fn read<R: Read + Seek>(
        reader: &mut Context<R>,
        t: &PropertyType,
        size: u64,
        count: u32,
    ) -> TResult<ValueVec> {
        Ok(match t {
            PropertyType::IntProperty => {
                ValueVec::Int(read_array(count, reader, |r| Ok(r.read_i32::<LE>()?))?)
            }
            PropertyType::Int16Property => {
                ValueVec::Int16(read_array(count, reader, |r| Ok(r.read_i16::<LE>()?))?)
            }
            PropertyType::Int64Property => {
                ValueVec::Int64(read_array(count, reader, |r| Ok(r.read_i64::<LE>()?))?)
            }
            PropertyType::UInt16Property => {
                ValueVec::UInt16(read_array(count, reader, |r| Ok(r.read_u16::<LE>()?))?)
            }
            PropertyType::UInt32Property => {
                ValueVec::UInt32(read_array(count, reader, |r| Ok(r.read_u32::<LE>()?))?)
            }
            PropertyType::FloatProperty => {
                ValueVec::Float(read_array(count, reader, |r| Ok(r.read_f32::<LE>()?))?)
            }
            PropertyType::DoubleProperty => {
                ValueVec::Double(read_array(count, reader, |r| Ok(r.read_f64::<LE>()?))?)
            }
            PropertyType::BoolProperty => {
                ValueVec::Bool(read_array(count, reader, |r| Ok(r.read_u8()? > 0))?)
            }
            PropertyType::ByteProperty => {
                if size == (count as u64) {
                    ValueVec::Byte(ByteArray::Byte(read_array(count, reader, |r| {
                        Ok(r.read_u8()?)
                    })?))
                } else {
                    ValueVec::Byte(ByteArray::Label(read_array(count, reader, |r| {
                        read_string(r)
                    })?))
                }
            }
            PropertyType::EnumProperty => {
                ValueVec::Enum(read_array(count, reader, |r| read_string(r))?)
            }
            PropertyType::StrProperty => ValueVec::Str(read_array(count, reader, read_string)?),
            PropertyType::TextProperty => ValueVec::Text(read_array(count, reader, Text::read)?),
            PropertyType::SoftObjectProperty => {
                ValueVec::SoftObject(read_array(count, reader, |r| {
                    Ok((read_string(r)?, read_string(r)?))
                })?)
            }
            PropertyType::NameProperty => ValueVec::Name(read_array(count, reader, read_string)?),
            PropertyType::ObjectProperty => {
                ValueVec::Object(read_array(count, reader, read_string)?)
            }
            _ => return Err(Error::UnknownVecType(format!("{t:?}"))),
        })
    }
}
impl ValueArray {
    fn read<R: Read + Seek>(
        reader: &mut Context<R>,
        t: &PropertyType,
        size: u64,
        progress: &Option<Function>
    ) -> TResult<ValueArray> {
        let count = reader.read_u32::<LE>()?;
        Ok(match t {
            PropertyType::StructProperty => {
                let _type = read_string(reader)?;
                let name = read_string(reader)?;
                let _size = reader.read_u64::<LE>()?;
                let struct_type = StructType::read(reader)?;
                let id = uuid::Uuid::read(reader)?.to_string();
                reader.read_u8()?;
                let mut value = vec![];
                for _ in 0..count {
                    value.push(StructValue::read(reader, &struct_type, progress)?);
                }
                ValueArray::Struct {
                    _type,
                    name,
                    struct_type,
                    id,
                    value,
                }
            }
            _ => ValueArray::Base(ValueVec::read(reader, t, size, count)?),
        })
    }
}
impl ValueSet {
    fn read<R: Read + Seek>(
        reader: &mut Context<R>,
        t: &PropertyType,
        st: Option<&StructType>,
        size: u64,
        progress: &Option<Function>
    ) -> TResult<ValueSet> {
        let count = reader.read_u32::<LE>()?;
        Ok(match t {
            PropertyType::StructProperty => ValueSet::Struct(read_array(count, reader, |r| {
                StructValue::read(r, st.unwrap(), progress)
            })?),
            _ => ValueSet::Base(ValueVec::read(reader, t, size, count)?),
        })
    }
}

/// Properties consist of an ID and a value and are present in [`Root`] and [`StructValue::Struct`]
#[derive(Debug, PartialEq)]
pub enum Property {
    Int8 {
        id: Option<String>,
        value: Int8,
    },
    Int16 {
        id: Option<String>,
        value: Int16,
    },
    Int {
        id: Option<String>,
        value: Int,
    },
    Int64 {
        id: Option<String>,
        value: Int64,
    },
    UInt8 {
        id: Option<String>,
        value: UInt8,
    },
    UInt16 {
        id: Option<String>,
        value: UInt16,
    },
    UInt32 {
        id: Option<String>,
        value: UInt32,
    },
    UInt64 {
        id: Option<String>,
        value: UInt64,
    },
    Float {
        id: Option<String>,
        value: Float,
    },
    Double {
        id: Option<String>,
        value: Double,
    },
    Bool {
        id: Option<String>,
        value: Bool,
    },
    Byte {
        id: Option<String>,
        value: Byte,
        enum_type: String,
    },
    Enum {
        id: Option<String>,
        value: Enum,
        enum_type: String,
    },
    Str {
        id: Option<String>,
        value: String,
    },
    FieldPath {
        id: Option<String>,
        value: FieldPath,
    },
    SoftObject {
        id: Option<String>,
        value: String,
        value2: String,
    },
    Name {
        id: Option<String>,
        value: String,
    },
    Object {
        id: Option<String>,
        value: String,
    },
    Text {
        id: Option<String>,
        value: Text,
    },
    Delegate {
        id: Option<String>,
        value: Delegate,
    },
    MulticastDelegate {
        id: Option<String>,
        value: MulticastDelegate,
    },
    MulticastInlineDelegate {
        id: Option<String>,
        value: MulticastInlineDelegate,
    },
    MulticastSparseDelegate {
        id: Option<String>,
        value: MulticastSparseDelegate,
    },
    Set {
        id: Option<String>,
        set_type: PropertyType,
        value: ValueSet,
    },
    Map {
        id: Option<String>,
        key_type: PropertyType,
        value_type: PropertyType,
        value: Vec<MapEntry>,
    },
    Struct {
        id: Option<String>,
        value: StructValue,
        struct_type: StructType,
        struct_id: uuid::Uuid,
    },
    RawData {
        id: Option<String>,
        properties: Properties,
        struct_id: uuid::Uuid,
    },
    Array {
        array_type: PropertyType,
        id: Option<String>,
        value: ValueArray,
    },
}
impl From<Property> for JsValue {
    fn from(prop: Property) -> Self {
        match prop {
            Property::Int8 { value, .. } => JsValue::from(value),
            Property::Int16 { value, .. } => JsValue::from(value),
            Property::Int { value, .. } => JsValue::from(value),
            Property::Int64 { value, .. } => JsValue::from(value),
            Property::UInt8 { value, .. } => JsValue::from(value),
            Property::UInt16 { value, .. } => JsValue::from(value),
            Property::UInt32 { value, .. } => JsValue::from(value),
            Property::UInt64 { value, .. } => JsValue::from(value),
            Property::Float { value, .. } => JsValue::from(value),
            Property::Double { value, .. } => JsValue::from(value),
            Property::Bool { value, .. } => JsValue::from(value),
            Property::Enum { value, .. } => JsValue::from(value),
            Property::Name { value, .. } => JsValue::from(value),
            Property::Str { value, .. } => JsValue::from(value),
            Property::Object { value, .. } => JsValue::from(value),
            Property::Byte { value, .. } => JsValue::from(value),

            Property::SoftObject { value, value2, .. } => {
                // String, String
                let ary = Array::new_with_length(2);
                ary.set(0, JsValue::from(value));
                ary.set(1, JsValue::from(value2));
                return ary.into();
            },

            Property::Text { value, .. } => {
                // uesave::Text
                match value.variant {
                    TextVariant::None { .. } => JsValue::null(),
                    TextVariant::Base { source_string, .. } => JsString::from(source_string).into(),
                    TextVariant::AsDate { source_date_time, .. } => JsValue::from(source_date_time),
                    TextVariant::ArgumentFormat { .. } => JsValue::null(),
                    TextVariant::AsNumber { source_value, .. } => {
                        match source_value {
                            FFormatArgumentValue::Int(i) => JsValue::from(i),
                            FFormatArgumentValue::UInt(ui) => JsValue::from(ui),
                            FFormatArgumentValue::Float(f) => JsValue::from(f),
                            FFormatArgumentValue::Double(d) => JsValue::from(d),
                            FFormatArgumentValue::Text(_) => JsValue::null(),
                            FFormatArgumentValue::Gender(g) => JsValue::from(g),
                        }
                    },
                    TextVariant::StringTableEntry { table, key } => {
                        JsString::from(key)
                            .concat(&JsString::from(table).into())
                            .into()
                    }
                }
            },

            Property::Delegate { value, .. } => {
                // uesave::Delegate
                // name: String, path: String
                JsString::from(value.path)
                    .concat(&JsString::from("/").into())
                    .concat(&JsString::from(value.name).into())
                    .into()
            },

            Property::Set { value, .. } => JsValue::from(value),
            Property::Struct { value, .. } => JsValue::from(value),
            Property::Array { value, .. } => JsValue::from(value),

            Property::Map { key_type, value_type, value, .. } => {
                let map = Map::new();
                map.set(&JsString::from("KeyType").into(), &JsString::from(key_type.get_name()).into());
                map.set(&JsString::from("ValueType").into(), &JsString::from(value_type.get_name()).into());

                let array = Array::new();
                for entry in value {
                    let map = Map::new();
                    map.set(&JsString::from("Key").into(), &JsValue::from(entry.key));
                    map.set(&JsString::from("Value").into(), &JsValue::from(entry.value));
                    array.push(&map_to_object(&map));
                }
                
                map.set(&JsString::from("Values").into(), &array.into());
                map_to_object(&map)
            },
            Property::RawData { properties, .. } => map_to_object(&properties),
            _ => JsValue::null()
        }
    }
}
impl Property {
    fn read<R: Read + Seek>(
        reader: &mut Context<R>,
        t: PropertyType,
        size: u64,
        progress: &Option<Function>
    ) -> TResult<Property> {
        match t {
            PropertyType::Int8Property => Ok(Property::Int8 {
                id: read_optional_uuid(reader)?,
                value: reader.read_i8()?,
            }),
            PropertyType::Int16Property => Ok(Property::Int16 {
                id: read_optional_uuid(reader)?,
                value: reader.read_i16::<LE>()?,
            }),
            PropertyType::IntProperty => Ok(Property::Int {
                id: read_optional_uuid(reader)?,
                value: reader.read_i32::<LE>()?,
            }),
            PropertyType::Int64Property => Ok(Property::Int64 {
                id: read_optional_uuid(reader)?,
                value: reader.read_i64::<LE>()?,
            }),
            PropertyType::UInt8Property => Ok(Property::UInt8 {
                id: read_optional_uuid(reader)?,
                value: reader.read_u8()?,
            }),
            PropertyType::UInt16Property => Ok(Property::UInt16 {
                id: read_optional_uuid(reader)?,
                value: reader.read_u16::<LE>()?,
            }),
            PropertyType::UInt32Property => Ok(Property::UInt32 {
                id: read_optional_uuid(reader)?,
                value: reader.read_u32::<LE>()?,
            }),
            PropertyType::UInt64Property => Ok(Property::UInt64 {
                id: read_optional_uuid(reader)?,
                value: reader.read_u64::<LE>()?,
            }),
            PropertyType::FloatProperty => Ok(Property::Float {
                id: read_optional_uuid(reader)?,
                value: reader.read_f32::<LE>()?,
            }),
            PropertyType::DoubleProperty => Ok(Property::Double {
                id: read_optional_uuid(reader)?,
                value: reader.read_f64::<LE>()?,
            }),
            PropertyType::BoolProperty => Ok(Property::Bool {
                value: reader.read_u8()? > 0,
                id: read_optional_uuid(reader)?,
            }),
            PropertyType::ByteProperty => Ok({
                let enum_type = read_string(reader)?;
                let id = read_optional_uuid(reader)?;
                let value = if enum_type == "None" {
                    Byte::Byte(reader.read_u8()?)
                } else {
                    Byte::Label(read_string(reader)?)
                };
                Property::Byte {
                    enum_type,
                    id,
                    value,
                }
            }),
            PropertyType::EnumProperty => Ok(Property::Enum {
                enum_type: read_string(reader)?,
                id: read_optional_uuid(reader)?,
                value: read_string(reader)?,
            }),
            PropertyType::NameProperty => Ok(Property::Name {
                id: read_optional_uuid(reader)?,
                value: read_string(reader)?,
            }),
            PropertyType::StrProperty => Ok(Property::Str {
                id: read_optional_uuid(reader)?,
                value: read_string(reader)?,
            }),
            PropertyType::FieldPathProperty => Ok(Property::FieldPath {
                id: read_optional_uuid(reader)?,
                value: FieldPath::read(reader)?,
            }),
            PropertyType::SoftObjectProperty => Ok(Property::SoftObject {
                id: read_optional_uuid(reader)?,
                value: read_string(reader)?,
                value2: read_string(reader)?,
            }),
            PropertyType::ObjectProperty => Ok(Property::Object {
                id: read_optional_uuid(reader)?,
                value: read_string(reader)?,
            }),
            PropertyType::TextProperty => Ok(Property::Text {
                id: read_optional_uuid(reader)?,
                value: Text::read(reader)?,
            }),
            PropertyType::DelegateProperty => Ok(Property::Delegate {
                id: read_optional_uuid(reader)?,
                value: Delegate::read(reader)?,
            }),
            PropertyType::MulticastDelegateProperty => Ok(Property::MulticastDelegate {
                id: read_optional_uuid(reader)?,
                value: MulticastDelegate::read(reader)?,
            }),
            PropertyType::MulticastInlineDelegateProperty => {
                Ok(Property::MulticastInlineDelegate {
                    id: read_optional_uuid(reader)?,
                    value: MulticastInlineDelegate::read(reader)?,
                })
            }
            PropertyType::MulticastSparseDelegateProperty => {
                Ok(Property::MulticastSparseDelegate {
                    id: read_optional_uuid(reader)?,
                    value: MulticastSparseDelegate::read(reader)?,
                })
            }
            PropertyType::SetProperty => {
                let set_type = PropertyType::read(reader, progress)?;
                let id = read_optional_uuid(reader)?;
                reader.read_u32::<LE>()?;
                let struct_type = match set_type {
                    PropertyType::StructProperty => Some(reader.get_type_or(&StructType::Guid)?),
                    _ => None,
                };
                let value = ValueSet::read(reader, &set_type, struct_type, size - 8, progress)?;
                Ok(Property::Set {
                    id,
                    set_type,
                    value,
                })
            }
            PropertyType::MapProperty => {
                let key_type = PropertyType::read(reader, &None)?;
                let value_type = PropertyType::read(reader, &None)?;
                let id = read_optional_uuid(reader)?;
                reader.read_u32::<LE>()?;
                let count = reader.read_u32::<LE>()?;
                let mut value = vec![];

                let key_struct_type = match key_type {
                    PropertyType::StructProperty => {
                        Some(reader.scope("Key", |r| r.get_type_or(&StructType::Guid))?)
                    }
                    _ => None,
                };
                let value_struct_type = match value_type {
                    PropertyType::StructProperty => {
                        Some(reader.scope("Value", |r| r.get_type_or(&StructType::Struct(None)))?)
                    }
                    _ => None,
                };

                for _ in 0..count {
                    value.push(MapEntry::read(
                        reader,
                        &key_type,
                        key_struct_type,
                        &value_type,
                        value_struct_type,
                        progress
                    )?)
                }

                Ok(Property::Map {
                    key_type,
                    value_type,
                    id,
                    value,
                })
            }
            PropertyType::StructProperty => {
                log("Property was a struct");
                let struct_type = StructType::read(reader)?;
                let struct_id = uuid::Uuid::read(reader)?;
                let id = read_optional_uuid(reader)?;
                let value = StructValue::read(reader, &struct_type, progress)?;
                Ok(Property::Struct {
                    struct_type,
                    struct_id,
                    id,
                    value,
                })
            }
            PropertyType::ArrayProperty => {
                let array_type = PropertyType::read(reader, &None)?;
                let id: Option<String> = read_optional_uuid(reader)?;
                let value = ValueArray::read(reader, &array_type, size - 4, progress)?;

                Ok(Property::Array {
                    array_type,
                    id,
                    value,
                })
            }
        }
    }
}

#[wasm_bindgen(getter_with_clone)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CustomFormatData {
    pub id: String,
    pub value: i32,
}

impl<R: Read + Seek> Readable<R> for CustomFormatData {
    fn read(reader: &mut Context<R>) -> TResult<Self> {
        Ok(CustomFormatData {
            id: uuid::Uuid::read(reader)?.to_string(),
            value: reader.read_i32::<LE>()?,
        })
    }
}

#[wasm_bindgen(getter_with_clone)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    pub magic: u32,
    pub save_game_version: u32,
    pub package_version: u64,
    pub engine_version_major: u16,
    pub engine_version_minor: u16,
    pub engine_version_patch: u16,
    pub engine_version_build: u32,
    pub engine_version: String,
    pub custom_format_version: u32,
    pub custom_format: Vec<CustomFormatData>,
}
impl Header {
    fn large_world_coordinates(&self) -> bool {
        self.engine_version_major >= 5
    }
}
impl<R: Read + Seek> Readable<R> for Header {
    fn read(reader: &mut Context<R>) -> TResult<Self> {
        let magic = reader.read_u32::<LE>()?;
        if magic != u32::from_le_bytes(*b"GVAS") {
            eprintln!(
                "Found non-standard magic: {:02x?} ({}) expected: GVAS, continuing to parse...",
                &magic.to_le_bytes(),
                String::from_utf8_lossy(&magic.to_le_bytes())
            );
        }
        let save_game_version = reader.read_u32::<LE>()?;
        let package_version = if save_game_version < 3 {
            reader.read_u32::<LE>()? as u64
        } else {
            reader.read_u64::<LE>()?
        };
        Ok(Header {
            magic,
            save_game_version,
            package_version,
            engine_version_major: reader.read_u16::<LE>()?,
            engine_version_minor: reader.read_u16::<LE>()?,
            engine_version_patch: reader.read_u16::<LE>()?,
            engine_version_build: reader.read_u32::<LE>()?,
            engine_version: read_string(reader)?,
            custom_format_version: reader.read_u32::<LE>()?,
            custom_format: read_array(reader.read_u32::<LE>()?, reader, CustomFormatData::read)?,
        })
    }
}

/// Root struct inside a save file which holds both the Unreal Engine class name and list of properties
#[wasm_bindgen(getter_with_clone)]
#[derive(Debug, Clone, PartialEq)]
pub struct Root {
    #[wasm_bindgen(js_name="SaveGameType")]
    pub save_game_type: String,
    
    #[wasm_bindgen(js_name="worldSaveData")]
    pub world_save_data: Properties,
}
impl Root {
    fn read<R: Read + Seek>(reader: &mut Context<R>, progress: &Option<Function>) -> TResult<Self> {
        Ok(Self {
            save_game_type: read_string(reader)?,
            world_save_data: read_properties_until_none(reader, progress)?,
        })
    }
}

fn vec_to_array<T>(vec: Vec<T>) -> Array
where T: Into<JsValue>
{
    let array = Array::new_with_length(vec.len() as u32);
    for i in vec {
        array.push(&i.into());
    }

    return array;
}

fn tuple_vec_to_array<T>(vec: Vec<(T, T)>) -> Array
where T: Into<JsValue>
{
    let array = Array::new_with_length(vec.len() as u32);
    for (a, b) in vec {
        let tup = Array::new_with_length(2);
        tup.set(0, a.into());
        tup.set(1, b.into());
        array.push(&tup);
    }

    return array;
}

#[wasm_bindgen(getter_with_clone)]
#[derive(Debug, PartialEq)]
pub struct Save {
    pub header: Header,
    pub root: Root,
    pub extra: Array,
}
impl Save {
    /// Reads save from the given reader
    pub fn read<R: Read>(reader: &mut R, progress: &Option<Function>) -> Result<Self, ParseError> {
        Self::read_with_types(reader, &Types::new(), progress)
    }
    /// Reads save from the given reader using the provided [`Types`]
    pub fn read_with_types<R: Read>(reader: &mut R, types: &Types, progress: &Option<Function>) -> Result<Self, ParseError> {
        let mut reader = SeekReader::new(reader);

        Context::run_with_types(types, &mut reader, |reader| {
            let header = Header::read(reader)?;
            report_progress(reader, progress);

            let (root, extra) = reader.header(&header, |reader| -> TResult<_> {
                let root = Root::read(reader, progress)?;
                let extra = {
                    let mut buf = vec![];
                    reader.read_to_end(&mut buf)?;
                    if buf != [0; 4] {
                        eprintln!(
                            "{} extra bytes. Save may not have been parsed completely.",
                            buf.len()
                        );
                    }
                    buf
                };

                Ok((root, vec_to_array(extra)))
            })?;

            Ok(Self {
                header,
                root,
                extra,
            })
        })
        .map_err(|e| error::ParseError {
            offset: reader.stream_position().unwrap() as usize, // our own implemenation which cannot fail
            error: e,
        })
    }
}

impl From<error::Error> for JsValue {
    fn from(value: error::Error) -> Self {
        value.to_string().into()
    }
}

#[wasm_bindgen]
extern {
    #[wasm_bindgen(js_namespace = console)]
    fn error(s: &str);

    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);

}

// When initially parsing, the ValueArray of the Pal data is a ByteProperty
// meaning it split each pal into it's own respective byte data, but they are not deserialized yet
// this can take one element and deserialize it...
#[wasm_bindgen(js_name="palFromRaw")]
pub fn pal_from_raw(buffer: &Uint8Array, types: Option<Map>, progress: Option<Function>) -> Result<JsValue, JsValue> {
    let buf: Vec<u8> = buffer.to_vec();
    let file = Cursor::new(&buf);
    let mut reader = SeekReader::new(file);

    let mut type_map = Types::new();
    if types.is_some() {
        types.unwrap().for_each(&mut |value, key| {
            log(("adding type ".to_owned() + &key.as_string().unwrap()).as_str());
            type_map.add(key.as_string().unwrap(), StructType::Struct(value.as_string()));
        });
    }

    let read_result = Context::run_with_types(&type_map, &mut reader, |reader| {
        read_property(reader, &progress)
    })?;
    
    match read_result {
        Some((_, property)) => Ok(property.into()),
        None => Err(JsValue::from("Could not convert pal from provided data."))
    }
}

#[wasm_bindgen]
pub fn deserialize(buffer: &Uint8Array, types: Option<Map>, progress: Option<Function>) -> Result<Save, JsValue> {
    utils::set_panic_hook();

    let mut type_map = Types::new();
    if types.is_some() {
        types.unwrap().for_each(&mut |value, key| {
            type_map.add(key.as_string().unwrap(), StructType::Struct(value.as_string()));
        });
    }

    let buf: Vec<u8> = buffer.to_vec();
    let mut file = Cursor::new(&buf);

    match Save::read_with_types(&mut file, &type_map, &progress) {
        Ok(save) => Ok(save),
        Err(err) => {
            error("Read save failed");
            let s = err.to_string();
            error(&s);
            panic!("{}", s);
        }
    }
}