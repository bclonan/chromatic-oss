//! Standalone native color tensor encoder, decoder and arithmetic driver.

use hue_tensor::storage::{load_tensor, save_tensor};
use hue_tensor::{Error, Result, Tensor, Token, MODULUS};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_MAX_BYTES: usize = 512 * 1024 * 1024;
static DEMO_COUNTER: AtomicU64 = AtomicU64::new(0);

const USAGE: &str = "hue-tensor-cli demo [OUTPUT]\n\
hue-tensor-cli encode SHAPE PHASES\n\
hue-tensor-cli decode SHAPE RGBD\n\
hue-tensor-cli save OUTPUT SHAPE PHASES\n\
hue-tensor-cli read INPUT [MAX_BYTES]\n\
hue-tensor-cli add|sub|product|matmul LEFT RIGHT OUTPUT\n\
hue-tensor-cli rotate INPUT OUTPUT TICKS\n\
hue-tensor-cli blend LEFT RIGHT OUTPUT WEIGHT\n\
hue-tensor-cli sum INPUT OUTPUT AXIS\n\n\
SHAPE is scalar or dimensions such as 2x2. PHASES are comma-separated\n\
integer ticks in 0..391680. RGBD rows use r,g,b,d;r,g,b,d. Use - for\n\
an empty token list. All arithmetic uses the finite hue domain.";

fn invalid(message: impl Into<String>) -> Error {
    Error(message.into())
}

fn arity(arguments: &[String], expected: usize) -> Result<()> {
    if arguments.len() == expected {
        Ok(())
    } else {
        Err(invalid(format!(
            "expected {} command arguments, received {}\n{USAGE}",
            expected.saturating_sub(1),
            arguments.len().saturating_sub(1)
        )))
    }
}

fn shape(text: &str) -> Result<Vec<usize>> {
    if text == "scalar" {
        return Ok(Vec::new());
    }
    if text.is_empty() {
        return Err(invalid("shape cannot be empty; use scalar for rank zero"));
    }
    text.split('x')
        .map(|part| {
            part.parse::<usize>()
                .map_err(|_| invalid(format!("invalid shape dimension {part:?}")))
        })
        .collect()
}

fn phases(text: &str) -> Result<Vec<u32>> {
    if text == "-" {
        return Ok(Vec::new());
    }
    text.split(',')
        .map(|part| {
            part.parse::<u32>()
                .map_err(|_| invalid(format!("invalid phase tick {part:?}")))
        })
        .collect()
}

fn color_tokens(text: &str) -> Result<Vec<Token>> {
    if text == "-" {
        return Ok(Vec::new());
    }
    text.split(';')
        .map(|row| {
            let bytes: Vec<u8> = row
                .split(',')
                .map(|part| {
                    part.parse::<u8>()
                        .map_err(|_| invalid(format!("invalid color byte {part:?}")))
                })
                .collect::<Result<_>>()?;
            if bytes.len() != 4 {
                return Err(invalid("each color token must contain exactly R,G,B,D"));
            }
            Token::new(bytes[0], bytes[1], bytes[2], bytes[3])
        })
        .collect()
}

fn integers<T: std::fmt::Display>(values: &[T]) -> String {
    let mut result = String::from("[");
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            result.push(',');
        }
        result.push_str(&value.to_string());
    }
    result.push(']');
    result
}

fn token_json(token: &Token) -> String {
    integers(&[token.r, token.g, token.b, token.d])
}

fn tensor_json(tensor: &Tensor) -> String {
    let mut rows = String::from("[");
    for (index, token) in tensor.tokens().iter().enumerate() {
        if index != 0 {
            rows.push(',');
        }
        rows.push_str(&token_json(token));
    }
    rows.push(']');
    format!(
        "{{\"format\":\"hue-rgbd-v1\",\"modulus\":{MODULUS},\"shape\":{},\"token_count\":{},\"rgbd\":{rows},\"phases\":{}}}",
        integers(tensor.shape()), tensor.len(), integers(&tensor.phases())
    )
}

fn json_string(text: &str) -> String {
    let mut result = String::from("\"");
    for character in text.chars() {
        match character {
            '\"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            character if character < ' ' => {
                result.push_str(&format!("\\u{:04x}", character as u32));
            }
            character => result.push(character),
        }
    }
    result.push('\"');
    result
}

fn parse_number<T: std::str::FromStr>(text: &str, name: &str) -> Result<T> {
    text.parse::<T>()
        .map_err(|_| invalid(format!("invalid {name} {text:?}")))
}

fn write_and_print(tensor: &Tensor, path: &Path) -> Result<()> {
    save_tensor(tensor, path)?;
    println!("{}", tensor_json(tensor));
    Ok(())
}

fn load(path: &str) -> Result<Tensor> {
    load_tensor(Path::new(path), DEFAULT_MAX_BYTES)
}

fn matrix_reference(a: &[u32; 4], b: &[u32; 4]) -> Vec<u32> {
    let mut result = vec![0u32; 4];
    for row in 0..2 {
        for column in 0..2 {
            let mut accumulator = 0u64;
            for inner in 0..2 {
                accumulator += u64::from(a[row * 2 + inner]) * u64::from(b[inner * 2 + column]);
            }
            result[row * 2 + column] = (accumulator % u64::from(MODULUS)) as u32;
        }
    }
    result
}

fn demo(output: Option<&str>) -> Result<()> {
    let a = [256u32, 65_536, 131_072, 196_608];
    let b = [32_768u32, 98_304, 262_144, 390_000];
    let left = Tensor::from_phases(a.to_vec(), vec![2, 2])?;
    let right = Tensor::from_phases(b.to_vec(), vec![2, 2])?;
    let product = left.matmul(&right)?;
    let expected = matrix_reference(&a, &b);
    if product.phases() != expected {
        return Err(invalid(
            "native matrix multiplication differs from the independent phase oracle",
        ));
    }
    let old_token = left.tokens()[0];
    let mut actual_color_bytes = left.tokens().to_vec();
    actual_color_bytes[0].d = actual_color_bytes[0].d.wrapping_add(1);
    let changed = Tensor::from_rgbd(actual_color_bytes, left.shape().to_vec())?;
    let changed_product = changed.matmul(&right)?;
    let mut changed_a = a;
    changed_a[0] += 1;
    let changed_expected = matrix_reference(&changed_a, &b);
    if changed_product.phases() != changed_expected || changed_product.tokens() == product.tokens()
    {
        return Err(invalid(
            "changing the stored D byte did not produce the expected changed arithmetic",
        ));
    }
    let is_temporary = output.is_none();
    let output_path = match output {
        Some(path) => Path::new(path).to_path_buf(),
        None => {
            let instant = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|_| invalid("system clock precedes the Unix epoch"))?
                .as_nanos();
            let counter = DEMO_COUNTER.fetch_add(1, Ordering::Relaxed);
            std::env::temp_dir().join(format!(
                "hue-demo-{}-{instant}-{counter}.hue",
                std::process::id()
            ))
        }
    };
    let persisted = (|| -> Result<usize> {
        save_tensor(&changed_product, &output_path)?;
        let decoded = load_tensor(&output_path, DEFAULT_MAX_BYTES)?;
        if decoded != changed_product {
            return Err(invalid(
                "saved color tensor did not round-trip its exact bytes",
            ));
        }
        let bytes = std::fs::metadata(&output_path)
            .map_err(|error| invalid(format!("read saved tensor size: {error}")))?
            .len();
        usize::try_from(bytes).map_err(|_| invalid("saved tensor size exceeds the address space"))
    })();
    if is_temporary {
        let _ = std::fs::remove_file(&output_path);
    }
    let stored_bytes = persisted?;
    println!(
        "{{\"format\":\"hue-native-demo-v1\",\"modulus\":{MODULUS},\"left\":{},\"right\":{},\"matmul\":{},\"reference_phases\":{},\"mutation\":{{\"channel\":\"D\",\"before\":{},\"after\":{},\"before_phase\":{},\"after_phase\":{},\"changed_matmul\":{},\"reference_phases\":{}}},\"persistence\":{{\"path\":{},\"retained\":{},\"exact_roundtrip\":true,\"payload_bytes\":{},\"file_bytes\":{stored_bytes}}}}}",
        tensor_json(&left), tensor_json(&right), tensor_json(&product), integers(&expected),
        token_json(&old_token), token_json(&changed.tokens()[0]), old_token.phase(),
        changed.tokens()[0].phase(), tensor_json(&changed_product), integers(&changed_expected),
        json_string(&output_path.to_string_lossy()), !is_temporary, changed_product.len() * 4,
    );
    Ok(())
}

fn run(arguments: &[String]) -> Result<()> {
    let command = arguments.first().map(String::as_str).unwrap_or("help");
    match command {
        "help" | "--help" | "-h" => {
            println!("{USAGE}");
            Ok(())
        }
        "demo" => {
            if arguments.len() != 1 && arguments.len() != 2 {
                return Err(invalid(format!(
                    "demo takes an optional output path\n{USAGE}"
                )));
            }
            demo(arguments.get(1).map(String::as_str))
        }
        "encode" => {
            arity(arguments, 3)?;
            let tensor = Tensor::from_phases(phases(&arguments[2])?, shape(&arguments[1])?)?;
            println!("{}", tensor_json(&tensor));
            Ok(())
        }
        "decode" => {
            arity(arguments, 3)?;
            let tensor = Tensor::from_rgbd(color_tokens(&arguments[2])?, shape(&arguments[1])?)?;
            println!("{}", tensor_json(&tensor));
            Ok(())
        }
        "save" => {
            arity(arguments, 4)?;
            let tensor = Tensor::from_phases(phases(&arguments[3])?, shape(&arguments[2])?)?;
            write_and_print(&tensor, Path::new(&arguments[1]))
        }
        "read" => {
            if arguments.len() != 2 && arguments.len() != 3 {
                return Err(invalid(format!(
                    "read needs a path and optional byte budget\n{USAGE}"
                )));
            }
            let max_bytes = arguments
                .get(2)
                .map(|text| parse_number(text, "byte budget"))
                .transpose()?
                .unwrap_or(DEFAULT_MAX_BYTES);
            let tensor = load_tensor(Path::new(&arguments[1]), max_bytes)?;
            println!("{}", tensor_json(&tensor));
            Ok(())
        }
        "add" | "sub" | "product" | "matmul" => {
            arity(arguments, 4)?;
            let left = load(&arguments[1])?;
            let right = load(&arguments[2])?;
            let tensor = match command {
                "add" => left.add(&right)?,
                "sub" => left.sub(&right)?,
                "product" => left.product(&right)?,
                "matmul" => left.matmul(&right)?,
                _ => unreachable!(),
            };
            write_and_print(&tensor, Path::new(&arguments[3]))
        }
        "rotate" => {
            arity(arguments, 4)?;
            let tensor =
                load(&arguments[1])?.rotate(parse_number(&arguments[3], "rotation ticks")?)?;
            write_and_print(&tensor, Path::new(&arguments[2]))
        }
        "blend" => {
            arity(arguments, 5)?;
            let tensor = load(&arguments[1])?.blend(
                &load(&arguments[2])?,
                parse_number(&arguments[4], "blend weight")?,
            )?;
            write_and_print(&tensor, Path::new(&arguments[3]))
        }
        "sum" => {
            arity(arguments, 4)?;
            let tensor =
                load(&arguments[1])?.sum(parse_number(&arguments[3], "reduction axis")?)?;
            write_and_print(&tensor, Path::new(&arguments[2]))
        }
        _ => Err(invalid(format!("unknown command {command:?}\n{USAGE}"))),
    }
}

fn main() {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if let Err(error) = run(&arguments) {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_exact_phase_and_color_lists() {
        assert_eq!(shape("2x3").unwrap(), [2, 3]);
        assert_eq!(shape("scalar").unwrap(), Vec::<usize>::new());
        assert_eq!(phases("0,391679").unwrap(), [0, 391_679]);
        assert!(phases("1.5").is_err());
        assert!(phases("-1").is_err());
        let colors = color_tokens("255,0,0,0;255,1,0,255").unwrap();
        assert_eq!(colors[0].phase(), 0);
        assert_eq!(colors[1].phase(), 511);
        assert!(color_tokens("255,0,0").is_err());
        assert!(color_tokens("1,2,3,4").is_err());
        assert!(color_tokens("255,0,0,256").is_err());
    }

    #[test]
    fn escapes_json_strings_and_prints_actual_tokens() {
        assert_eq!(json_string("x\"\\\n\u{0001}"), "\"x\\\"\\\\\\n\\u0001\"");
        let tensor = Tensor::from_phases(vec![0, 256], vec![2]).unwrap();
        let json = tensor_json(&tensor);
        assert!(json.contains("\"rgbd\":[[255,0,0,0],[255,1,0,0]]"));
        assert!(json.contains("\"phases\":[0,256]"));
    }

    #[test]
    fn rejects_bad_commands_without_creating_files() {
        for arguments in [
            vec!["unknown"],
            vec!["save", "missing"],
            vec!["encode", "2", "0"],
            vec!["encode", "scalar", "391680"],
            vec!["decode", "scalar", "0,0,0,0"],
            vec!["read"],
        ] {
            let owned: Vec<String> = arguments.iter().map(|text| text.to_string()).collect();
            assert!(run(&owned).is_err(), "{arguments:?}");
        }
    }

    #[test]
    fn demo_checks_arithmetic_and_mutated_color_persistence() {
        demo(None).unwrap();
    }
}
