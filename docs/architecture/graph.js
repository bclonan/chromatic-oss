window.CHROMATIC_ARCHITECTURE = {
  "schema_version": 1,
  "groups": [
    {
      "id": "native",
      "name": "Native hue engine",
      "description": "Rust stores literal RGBD tokens and computes in Z/391680Z."
    },
    {
      "id": "entry",
      "name": "Native entry points",
      "description": "Python, C and the standalone CLI reach the same Rust engine."
    },
    {
      "id": "legacy",
      "name": "Legacy numerical APIs",
      "description": "PyTorch codecs, decoded operators and floating training remain explicit research APIs."
    },
    {
      "id": "dashboard",
      "name": "Local dashboard",
      "description": "A local HTTP server reads saved results and runs bounded legacy inspections."
    },
    {
      "id": "research",
      "name": "Experiments and evidence",
      "description": "Seeded datasets, measurements and source identities support research checks."
    },
    {
      "id": "release",
      "name": "Build and release",
      "description": "Offline Cargo builds, wheel assembly and acceptance checks produce local artifacts."
    }
  ],
  "nodes": [
    {
      "id": "tokens",
      "name": "RGBD token domain",
      "group": "native",
      "purpose": "Map canonical saturated RGB plus D to one of 391,680 integer phase ticks.",
      "inputs": "RGB bytes, canonical RGBD bytes or bounded integer ticks",
      "outputs": "Four-byte Token values or integer phases",
      "side_effects": "No I/O. General RGB projection discards saturation and brightness.",
      "ownership": "Rust Token value, copied into tensor storage",
      "trust_boundary": "Unvalidated colors and phases enter a finite domain",
      "failures": "Gray RGB, noncanonical RGBD or out-of-domain ticks fail.",
      "sources": [
        {
          "path": "native/hue_tensor/src/lib.rs",
          "symbol": "Token",
          "line": 35,
          "end_line": 35,
          "sha256": "ab8c42dbcd5750c20f16534c6adf97a3a11b501e898b0583a1fd897abc6c0ccb",
          "url": "source.html?path=native/hue_tensor/src/lib.rs#L35"
        },
        {
          "path": "native/hue_tensor/src/lib.rs",
          "symbol": "Token::from_rgb",
          "line": 104,
          "end_line": 104,
          "sha256": "ab8c42dbcd5750c20f16534c6adf97a3a11b501e898b0583a1fd897abc6c0ccb",
          "url": "source.html?path=native/hue_tensor/src/lib.rs#L104"
        }
      ],
      "tests": [
        {
          "path": "tests/test_native_hue.py",
          "symbol": "test_exact_color_corners_fine_channel_and_public_aliases",
          "line": 32,
          "end_line": 45,
          "sha256": "58d60cc357bf4734cc83a7a91d39b161919643229bc465433eeef0748e1ed8dd",
          "url": "source.html?path=tests/test_native_hue.py#L32"
        }
      ],
      "kind": "component",
      "dependencies": [],
      "status": "implemented"
    },
    {
      "id": "tensor",
      "name": "Owned Rust tensor",
      "group": "native",
      "purpose": "Own Vec<Token> storage and a logical shape. Rank and element budgets bound allocations.",
      "inputs": "Validated tokens and dimensions",
      "outputs": "Owned tensor with copied token bytes",
      "side_effects": "Allocates memory. No hidden real-number tensor.",
      "ownership": "Rust Tensor owns both vectors. Constructors create independent outputs.",
      "trust_boundary": "Shape/count validation precedes allocation",
      "failures": "Rank above 32, count above 67,108,864, overflow or mismatched shape fails.",
      "sources": [
        {
          "path": "native/hue_tensor/src/lib.rs",
          "symbol": "Tensor",
          "line": 133,
          "end_line": 133,
          "sha256": "ab8c42dbcd5750c20f16534c6adf97a3a11b501e898b0583a1fd897abc6c0ccb",
          "url": "source.html?path=native/hue_tensor/src/lib.rs#L133"
        },
        {
          "path": "native/hue_tensor/src/lib.rs",
          "symbol": "shape_len",
          "line": 138,
          "end_line": 138,
          "sha256": "ab8c42dbcd5750c20f16534c6adf97a3a11b501e898b0583a1fd897abc6c0ccb",
          "url": "source.html?path=native/hue_tensor/src/lib.rs#L138"
        }
      ],
      "tests": [
        {
          "path": "tests/test_native_hue.py",
          "symbol": "test_invalid_shapes_operations_and_closed_handles_fail",
          "line": 175,
          "end_line": 203,
          "sha256": "58d60cc357bf4734cc83a7a91d39b161919643229bc465433eeef0748e1ed8dd",
          "url": "source.html?path=tests/test_native_hue.py#L175"
        }
      ],
      "kind": "component",
      "dependencies": [],
      "status": "implemented"
    },
    {
      "id": "operators",
      "name": "Integer hue operators",
      "group": "native",
      "purpose": "Add, subtract and multiply modulo 391,680. Contract rank-two matrices, reduce axes and copy shape transforms.",
      "inputs": "Owned operands, axes, ticks or byte blend weight",
      "outputs": "New owned tensor",
      "side_effects": "Allocates output. Inputs remain unchanged.",
      "ownership": "Rust reads operands and owns outputs",
      "trust_boundary": "Integer phase labels do not imply real arithmetic or physical color mixing",
      "failures": "Incompatible broadcast, contraction, slice or axis fails. Division, GPU and autograd are absent.",
      "sources": [
        {
          "path": "native/hue_tensor/src/lib.rs",
          "symbol": "matmul",
          "line": 328,
          "end_line": 328,
          "sha256": "ab8c42dbcd5750c20f16534c6adf97a3a11b501e898b0583a1fd897abc6c0ccb",
          "url": "source.html?path=native/hue_tensor/src/lib.rs#L328"
        },
        {
          "path": "native/hue_tensor/src/lib.rs",
          "symbol": "blend",
          "line": 310,
          "end_line": 310,
          "sha256": "ab8c42dbcd5750c20f16534c6adf97a3a11b501e898b0583a1fd897abc6c0ccb",
          "url": "source.html?path=native/hue_tensor/src/lib.rs#L310"
        },
        {
          "path": "native/hue_tensor/src/lib.rs",
          "symbol": "slice",
          "line": 436,
          "end_line": 436,
          "sha256": "ab8c42dbcd5750c20f16534c6adf97a3a11b501e898b0583a1fd897abc6c0ccb",
          "url": "source.html?path=native/hue_tensor/src/lib.rs#L436"
        }
      ],
      "tests": [
        {
          "path": "tests/test_native_hue.py",
          "symbol": "test_random_modular_algebra_and_immutability",
          "line": 67,
          "end_line": 87,
          "sha256": "58d60cc357bf4734cc83a7a91d39b161919643229bc465433eeef0748e1ed8dd",
          "url": "source.html?path=tests/test_native_hue.py#L67"
        }
      ],
      "kind": "component",
      "dependencies": [
        "tokens"
      ],
      "status": "implemented"
    },
    {
      "id": "storage",
      "name": "Native file codec",
      "group": "native",
      "purpose": "Save literal RGBD with shape metadata and SHA-256. Load only after length, budget, checksum and color validation.",
      "inputs": "Tensor or file path and byte budget",
      "outputs": "HUECT file or new owned tensor",
      "side_effects": "Reads files. Writes a synced adjacent temporary file and renames it.",
      "ownership": "Rust loader owns loaded buffers. Writer cleans temporary files on failure.",
      "trust_boundary": "Untrusted file bytes enter memory. SHA-256 checks corruption, not author identity.",
      "failures": "Oversize, malformed, trailing, corrupt or noncanonical files fail. Rename rules depend on the OS.",
      "sources": [
        {
          "path": "native/hue_tensor/src/storage.rs",
          "symbol": "save_tensor",
          "line": 103,
          "end_line": 103,
          "sha256": "204dcf09f6f17e6f64b806aa157f9b1af207a0b3cd0c3ec78d6661adca11c137",
          "url": "source.html?path=native/hue_tensor/src/storage.rs#L103"
        },
        {
          "path": "native/hue_tensor/src/storage.rs",
          "symbol": "load_tensor",
          "line": 167,
          "end_line": 167,
          "sha256": "204dcf09f6f17e6f64b806aa157f9b1af207a0b3cd0c3ec78d6661adca11c137",
          "url": "source.html?path=native/hue_tensor/src/storage.rs#L167"
        }
      ],
      "tests": [
        {
          "path": "tests/test_native_hue.py",
          "symbol": "test_corrupt_or_adversarial_files_fail_even_after_rehashing",
          "line": 246,
          "end_line": 273,
          "sha256": "58d60cc357bf4734cc83a7a91d39b161919643229bc465433eeef0748e1ed8dd",
          "url": "source.html?path=tests/test_native_hue.py#L246"
        }
      ],
      "kind": "store",
      "dependencies": [],
      "status": "implemented"
    },
    {
      "id": "ffi",
      "name": "C ABI boundary",
      "group": "entry",
      "purpose": "Expose opaque handles, output capacities and status codes. Catch Rust panics at exported operations.",
      "inputs": "Caller pointers, handles, shapes, buffers and UTF-8 paths",
      "outputs": "Status 0, 1, 2 or 3, output handle, copied buffers and thread-local error",
      "side_effects": "Allocates and frees native handles. Storage calls access files.",
      "ownership": "C caller must free once and keep handles alive for every call",
      "trust_boundary": "Declared capacities are checked. Arbitrary pointer validity cannot be proven.",
      "failures": "Invalid inputs reject. Dangling, aliased or concurrently freed pointers violate the C contract.",
      "sources": [
        {
          "path": "native/hue_tensor/src/ffi.rs",
          "symbol": "boundary",
          "line": 19,
          "end_line": 19,
          "sha256": "cd4e88e8cdd9be41c9196d3f8e22baf82c376398290ddf0c7f0b1ecf2144f920",
          "url": "source.html?path=native/hue_tensor/src/ffi.rs#L19"
        },
        {
          "path": "native/hue_tensor/src/ffi.rs",
          "symbol": "ht_free",
          "line": 222,
          "end_line": 222,
          "sha256": "cd4e88e8cdd9be41c9196d3f8e22baf82c376398290ddf0c7f0b1ecf2144f920",
          "url": "source.html?path=native/hue_tensor/src/ffi.rs#L222"
        },
        {
          "path": "native/hue_tensor/src/ffi.rs",
          "symbol": "ht_load",
          "line": 511,
          "end_line": 511,
          "sha256": "cd4e88e8cdd9be41c9196d3f8e22baf82c376398290ddf0c7f0b1ecf2144f920",
          "url": "source.html?path=native/hue_tensor/src/ffi.rs#L511"
        }
      ],
      "tests": [
        {
          "path": "tests/test_native_hue.py",
          "symbol": "test_parallel_close_read_and_reversed_operand_calls_do_not_crash_or_deadlock",
          "line": 283,
          "end_line": 328,
          "sha256": "58d60cc357bf4734cc83a7a91d39b161919643229bc465433eeef0748e1ed8dd",
          "url": "source.html?path=tests/test_native_hue.py#L283"
        }
      ],
      "kind": "component",
      "dependencies": [
        "storage"
      ],
      "status": "implemented"
    },
    {
      "id": "python",
      "name": "HueTensor Python binding",
      "group": "entry",
      "purpose": "Load a checkout library or the separately installed native wheel through ctypes. Acquire operand locks in object order and manage one opaque handle per instance.",
      "inputs": "Python integer sequences, byte buffers and paths",
      "outputs": "HueTensor objects and copied RGBD, RGB or phases",
      "side_effects": "Loads a local DLL or shared library, allocates handles and accesses native files.",
      "ownership": "Instance owns a handle. close is idempotent. Context exit closes it.",
      "trust_boundary": "Python types and paths cross into C. Native ABI version and modulus must match.",
      "failures": "Missing or incompatible binary, NUL path, closed handle, invalid dtype or native rejection raises an exception.",
      "sources": [
        {
          "path": "chromatensor/hue.py",
          "symbol": "_library_path",
          "line": 30,
          "end_line": 42,
          "sha256": "9781800dbbce9893283390ee3b409027f95d48a5fcef01e48864ea5cb3126c97",
          "url": "source.html?path=chromatensor/hue.py#L30"
        },
        {
          "path": "chromatensor/hue.py",
          "symbol": "_library",
          "line": 46,
          "end_line": 78,
          "sha256": "9781800dbbce9893283390ee3b409027f95d48a5fcef01e48864ea5cb3126c97",
          "url": "source.html?path=chromatensor/hue.py#L46"
        },
        {
          "path": "chromatensor/hue.py",
          "symbol": "_owned",
          "line": 138,
          "end_line": 150,
          "sha256": "9781800dbbce9893283390ee3b409027f95d48a5fcef01e48864ea5cb3126c97",
          "url": "source.html?path=chromatensor/hue.py#L138"
        },
        {
          "path": "chromatensor/hue.py",
          "symbol": "HueTensor",
          "line": 153,
          "end_line": 397,
          "sha256": "9781800dbbce9893283390ee3b409027f95d48a5fcef01e48864ea5cb3126c97",
          "url": "source.html?path=chromatensor/hue.py#L153"
        }
      ],
      "tests": [
        {
          "path": "tests/test_native_hue.py",
          "symbol": "test_NUL_path_is_rejected_before_native_IO_and_does_not_overwrite_target",
          "line": 213,
          "end_line": 222,
          "sha256": "58d60cc357bf4734cc83a7a91d39b161919643229bc465433eeef0748e1ed8dd",
          "url": "source.html?path=tests/test_native_hue.py#L213"
        }
      ],
      "kind": "component",
      "dependencies": [
        "ffi"
      ],
      "status": "implemented"
    },
    {
      "id": "package",
      "name": "Public package names",
      "group": "entry",
      "purpose": "Export HueTensor, Tensor and ColorTensor. Resolve legacy names only when a caller requests them.",
      "inputs": "Python imports and requested attribute name",
      "outputs": "Native aliases or explicit legacy API",
      "side_effects": "Caches resolved legacy attributes in the module.",
      "ownership": "Python import machinery owns module state",
      "trust_boundary": "Choosing a legacy name loads PyTorch research code",
      "failures": "Unknown names raise AttributeError. Native execution is unavailable until a library is present.",
      "sources": [
        {
          "path": "chromatensor/__init__.py",
          "symbol": "__getattr__",
          "line": 19,
          "end_line": 24,
          "sha256": "5a0d9049c45cabb47f36f66a6800cd714d3b0ff8463f3a275f1eb9948be432c7",
          "url": "source.html?path=chromatensor/__init__.py#L19"
        }
      ],
      "tests": [
        {
          "path": "tests/test_native_hue.py",
          "symbol": "test_exact_color_corners_fine_channel_and_public_aliases",
          "line": 32,
          "end_line": 45,
          "sha256": "58d60cc357bf4734cc83a7a91d39b161919643229bc465433eeef0748e1ed8dd",
          "url": "source.html?path=tests/test_native_hue.py#L32"
        }
      ],
      "kind": "component",
      "dependencies": [
        "legacy-tensor",
        "python"
      ],
      "status": "implemented"
    },
    {
      "id": "cli",
      "name": "Standalone Rust CLI",
      "group": "entry",
      "purpose": "Parse encode, decode, save, read, arithmetic and demo commands without Python.",
      "inputs": "Command arguments and local native files",
      "outputs": "JSON output and native tensor files",
      "side_effects": "Reads and writes requested files. Prints output and errors.",
      "ownership": "CLI owns tensors until process exit",
      "trust_boundary": "Untrusted text and files cross into bounded Rust constructors and storage",
      "failures": "Bad arity, invalid numbers, dimensions, file limits or operations exit with an error.",
      "sources": [
        {
          "path": "native/hue_tensor/src/bin/hue_tensor.rs",
          "symbol": "run",
          "line": 236,
          "end_line": 236,
          "sha256": "46438f9dc38d96ab1703875a883c8993fccdf77430851bda22746a4c03d3650c",
          "url": "source.html?path=native/hue_tensor/src/bin/hue_tensor.rs#L236"
        },
        {
          "path": "native/hue_tensor/src/bin/hue_tensor.rs",
          "symbol": "demo",
          "line": 168,
          "end_line": 168,
          "sha256": "46438f9dc38d96ab1703875a883c8993fccdf77430851bda22746a4c03d3650c",
          "url": "source.html?path=native/hue_tensor/src/bin/hue_tensor.rs#L168"
        }
      ],
      "tests": [
        {
          "path": "scripts/verify_native_hue_release.py",
          "symbol": "validate_cli_demo",
          "line": 93,
          "end_line": 119,
          "sha256": "6117f5fdcdcb1019e7c92b9f4b8d4183498ff355da43e2a2477418c903a50b4e",
          "url": "source.html?path=scripts/verify_native_hue_release.py#L93"
        }
      ],
      "kind": "component",
      "dependencies": [
        "operators",
        "storage",
        "tensor"
      ],
      "status": "implemented"
    },
    {
      "id": "header",
      "name": "C caller contract",
      "group": "entry",
      "purpose": "Declare the exported ABI and its ownership, buffer and thread rules for C consumers.",
      "inputs": "Consumer compile and API calls",
      "outputs": "Opaque handle declarations and function signatures",
      "side_effects": "No runtime work in the header itself.",
      "ownership": "External C consumer owns its buffers and handle lifetimes",
      "trust_boundary": "Header documents obligations that a raw pointer ABI cannot enforce",
      "failures": "Misdeclared buffers, double free and concurrent free can cause undefined behavior.",
      "sources": [
        {
          "path": "native/hue_tensor/include/hue_tensor.h",
          "contains": "typedef struct ht_tensor",
          "line": 15,
          "end_line": 15,
          "sha256": "81eaefa1bc501b8de9f96875a5a1723799db2532e8ba7b07bf7d6cd0c63a3f6c",
          "url": "source.html?path=native/hue_tensor/include/hue_tensor.h#L15"
        }
      ],
      "tests": [
        {
          "path": "scripts/verify_native_hue_release.py",
          "symbol": "validate_native_commands",
          "line": 67,
          "end_line": 90,
          "sha256": "6117f5fdcdcb1019e7c92b9f4b8d4183498ff355da43e2a2477418c903a50b4e",
          "url": "source.html?path=scripts/verify_native_hue_release.py#L67"
        }
      ],
      "kind": "component",
      "dependencies": [],
      "status": "implemented"
    },
    {
      "id": "legacy-tensor",
      "name": "Legacy ChromaTensor",
      "group": "legacy",
      "purpose": "Store encoded PyTorch color tensors and decode real numerical values through a scalar codec.",
      "inputs": "PyTorch tensors, code tensors and ScalarCodec",
      "outputs": "Encoded tensor or floating decoded tensor",
      "side_effects": "Allocates PyTorch tensors. May use the selected device.",
      "ownership": "Python object owns PyTorch codes and codec state",
      "trust_boundary": "Legacy RGBD uses D as an IEEE exponent. It is incompatible with native D sub-hue bytes.",
      "failures": "Invalid code shape, dtype or codec configuration fails.",
      "sources": [
        {
          "path": "chromatensor/tensor.py",
          "symbol": "ChromaTensor",
          "line": 14,
          "end_line": 221,
          "sha256": "7daff1f6685c23d2713a2879f18480d1097d878af50c82487628486ac8f72ed8",
          "url": "source.html?path=chromatensor/tensor.py#L14"
        }
      ],
      "tests": [
        {
          "path": "tests/test_tensor.py",
          "line": 1,
          "end_line": 137,
          "sha256": "cdaffb1041bf3d47643b9a5c3dfe60962ea7624862094f333be0484f68526d90",
          "url": "source.html?path=tests/test_tensor.py#L1"
        }
      ],
      "kind": "component",
      "dependencies": [
        "codecs"
      ],
      "status": "experimental"
    },
    {
      "id": "codecs",
      "name": "Scalar codecs and vocabulary",
      "group": "legacy",
      "purpose": "Encode scalar values with RGB, RGBD, HSV, uniform or learned codebooks. Tokenizers preserve vocabulary metadata.",
      "inputs": "Real scalar tensors, quantization parameters or learned samples",
      "outputs": "Color codes, token IDs and decoded floats",
      "side_effects": "Fitting changes codebooks and allocates tensors.",
      "ownership": "Codec or vocabulary owns its configuration and learned state",
      "trust_boundary": "Finite values, code ranges and metadata govern decode semantics",
      "failures": "Invalid numerical values, codes, vocabulary IDs or parameter ranges fail.",
      "sources": [
        {
          "path": "chromatensor/codecs/base.py",
          "symbol": "ScalarCodec",
          "line": 79,
          "end_line": 95,
          "sha256": "51cf4214dcb731871ff218ff72d312569068ce13b4a0d70f0e2952651d68657b",
          "url": "source.html?path=chromatensor/codecs/base.py#L79"
        },
        {
          "path": "chromatensor/codecs/rgbd.py",
          "symbol": "RGBDCodec",
          "line": 9,
          "end_line": 48,
          "sha256": "0de8229189a88d4ef2806cf295a8f3bc1f11efe308f4aa1b166e944f9b273617",
          "url": "source.html?path=chromatensor/codecs/rgbd.py#L9"
        },
        {
          "path": "chromatensor/tokenizer/__init__.py",
          "symbol": "TokenSequence",
          "line": 87,
          "end_line": 183,
          "sha256": "a3b4a1b7203fc8d4c52fe9a1b03c3d9eee3dcc78d7d82891f447f9fc98273e72",
          "url": "source.html?path=chromatensor/tokenizer/__init__.py#L87"
        }
      ],
      "tests": [
        {
          "path": "tests/test_codecs.py",
          "line": 1,
          "end_line": 245,
          "sha256": "5b244cce5daf69e9a41e1b3f80a50cef45bef60ecba16482ff3c6a5b4b78c772",
          "url": "source.html?path=tests/test_codecs.py#L1"
        },
        {
          "path": "tests/test_rgbd_tokens.py",
          "line": 1,
          "end_line": 364,
          "sha256": "c219392c56391983f02da4ccb0e0e1053d8cd244af61af790b3d9183dee6b94f",
          "url": "source.html?path=tests/test_rgbd_tokens.py#L1"
        }
      ],
      "kind": "component",
      "dependencies": [],
      "status": "experimental"
    },
    {
      "id": "legacy-ops",
      "name": "Decoded PyTorch operators",
      "group": "legacy",
      "purpose": "Decode operands, run PyTorch arithmetic and encode the result.",
      "inputs": "Legacy ChromaTensor operands",
      "outputs": "Legacy encoded output or floating similarity",
      "side_effects": "Allocates decoded intermediates and output codes.",
      "ownership": "PyTorch manages buffers",
      "trust_boundary": "These operations execute floating point and do not call the native ring operators",
      "failures": "Wrong operand types and invalid PyTorch shapes fail.",
      "sources": [
        {
          "path": "chromatensor/operators/core.py",
          "symbol": "_decoded",
          "line": 10,
          "end_line": 14,
          "sha256": "5db3a66e45c15f764845c71d92053f1d60f0d6bd7c6632b0d3fcabf6d47c35a7",
          "url": "source.html?path=chromatensor/operators/core.py#L10"
        }
      ],
      "tests": [
        {
          "path": "tests/test_operators.py",
          "line": 1,
          "end_line": 65,
          "sha256": "990203f49d2e13bca02dd7f4455aa28e68baafe39aa63ab374b869246f7e45c3",
          "url": "source.html?path=tests/test_operators.py#L1"
        }
      ],
      "kind": "component",
      "dependencies": [
        "legacy-tensor"
      ],
      "status": "experimental"
    },
    {
      "id": "nn",
      "name": "Quantized neural layers",
      "group": "legacy",
      "purpose": "Quantize and reconstruct weights or activations around PyTorch layers with straight-through gradients.",
      "inputs": "Floating masters, activations, quantization configuration",
      "outputs": "Floating predictions, gradients and encoded state reports",
      "side_effects": "Training updates floating parameters. Device kernels run through PyTorch.",
      "ownership": "Modules own parameters and quantizer state",
      "trust_boundary": "Quantization-aware training does not establish integer or entirely chromatic execution",
      "failures": "Unsupported representation, invalid bit configuration or layer shape fails.",
      "sources": [
        {
          "path": "chromatensor/nn/layers.py",
          "symbol": "ChromaLinear",
          "line": 26,
          "end_line": 42,
          "sha256": "2570772713de51c263c5ffd653d61e0f535dcb3b1b9ffc8501ae6e7e9ccde194",
          "url": "source.html?path=chromatensor/nn/layers.py#L26"
        },
        {
          "path": "chromatensor/autograd/quantization.py",
          "symbol": "_StraightThrough",
          "line": 114,
          "end_line": 123,
          "sha256": "9524cd0af99950a2852a84cadbd29b8d374a089dce01f793cf903da9e72fe393",
          "url": "source.html?path=chromatensor/autograd/quantization.py#L114"
        },
        {
          "path": "chromatensor/autograd/quantization.py",
          "symbol": "ChromaQuantizer",
          "line": 126,
          "end_line": 215,
          "sha256": "9524cd0af99950a2852a84cadbd29b8d374a089dce01f793cf903da9e72fe393",
          "url": "source.html?path=chromatensor/autograd/quantization.py#L126"
        }
      ],
      "tests": [
        {
          "path": "tests/test_nn.py",
          "line": 1,
          "end_line": 418,
          "sha256": "d3908175692e1e973cd7c113fa151a3a1bb701e8c322383e511748994dae770c",
          "url": "source.html?path=tests/test_nn.py#L1"
        },
        {
          "path": "tests/test_rgbd_nn.py",
          "line": 1,
          "end_line": 253,
          "sha256": "a09d41ece470bed998f29acdf50931f91224b63401166d7746a81997211c0aa1",
          "url": "source.html?path=tests/test_rgbd_nn.py#L1"
        }
      ],
      "kind": "component",
      "dependencies": [],
      "status": "experimental"
    },
    {
      "id": "cubes",
      "name": "Cube research",
      "group": "legacy",
      "purpose": "Rearrange encoded cells in ChromaCube and reconstruct Fourier fields in HolographicCube.",
      "inputs": "Encoded cell grid, twists, masks or real fields",
      "outputs": "Rearranged cube, reconstructed field and metrics",
      "side_effects": "Allocates tensors and FFT intermediates.",
      "ownership": "Cube owns codes, identities, configuration and metadata",
      "trust_boundary": "Fourier computation uses floating and complex transforms. Fragment recovery depends on the declared experiment.",
      "failures": "Invalid geometry, masks, codes, finite scale or codec fails.",
      "sources": [
        {
          "path": "chromatensor/cube.py",
          "symbol": "ChromaCube",
          "line": 25,
          "end_line": 278,
          "sha256": "743086b6121e60630681e4958a65f9bd2195e71ef04520ef43ad8fc2dbd24451",
          "url": "source.html?path=chromatensor/cube.py#L25"
        },
        {
          "path": "chromatensor/holographic.py",
          "symbol": "HolographicCube",
          "line": 127,
          "end_line": 258,
          "sha256": "379630fa90845a4a1ad12282edbd7fbad90403ee6f12cc70f25231530e4ff997",
          "url": "source.html?path=chromatensor/holographic.py#L127"
        }
      ],
      "tests": [
        {
          "path": "tests/test_cube.py",
          "line": 1,
          "end_line": 260,
          "sha256": "ce56c5f43379a3ddcb0c5d84938ab26bdec665021fe2aac1a494d6dbc7b15320",
          "url": "source.html?path=tests/test_cube.py#L1"
        },
        {
          "path": "tests/test_holographic.py",
          "line": 1,
          "end_line": 291,
          "sha256": "b448ab2e0f04b18855b976f80bdf3fc382deb4747faa541c6de17b74f20a917f",
          "url": "source.html?path=tests/test_holographic.py#L1"
        }
      ],
      "kind": "component",
      "dependencies": [
        "legacy-tensor"
      ],
      "status": "experimental"
    },
    {
      "id": "legacy-storage",
      "name": "Legacy files and checkpoints",
      "group": "legacy",
      "purpose": "Serialize bounded color tensor files and model checkpoints. Decode archives without arbitrary pickle execution.",
      "inputs": "Legacy tensor, model state or archive and budgets",
      "outputs": "CTENSOR files, checkpoint archives and loaded legacy state",
      "side_effects": "Reads, decompresses and atomically writes local files.",
      "ownership": "Loader owns loaded tensors and metadata",
      "trust_boundary": "Archive entries, expanded lengths, JSON and NumPy payloads cross validation boundaries",
      "failures": "Traversal, oversized members, inconsistent shapes, corrupt checksums or incompatible metadata fail.",
      "sources": [
        {
          "path": "chromatensor/serialization/format.py",
          "symbol": "_open_archive",
          "line": 75,
          "end_line": 110,
          "sha256": "6b2694152b7a2d1face0eb6db891ac6effea9a0d06287657b63be97e9936e8b1",
          "url": "source.html?path=chromatensor/serialization/format.py#L75"
        },
        {
          "path": "chromatensor/serialization/format.py",
          "symbol": "load_chroma_checkpoint",
          "line": 312,
          "end_line": 363,
          "sha256": "6b2694152b7a2d1face0eb6db891ac6effea9a0d06287657b63be97e9936e8b1",
          "url": "source.html?path=chromatensor/serialization/format.py#L312"
        }
      ],
      "tests": [
        {
          "path": "tests/test_serialization.py",
          "line": 1,
          "end_line": 108,
          "sha256": "1a33cf704b47831070438c3d7d99ad5c268ccab3039ca73b43ed2a46cdc4ccd7",
          "url": "source.html?path=tests/test_serialization.py#L1"
        },
        {
          "path": "tests/test_file_limits.py",
          "line": 1,
          "end_line": 404,
          "sha256": "bf1ded7fb665bbabf7f8782f3770babeb2e3553ed8fa1a6a8b1826d36acbe1d5",
          "url": "source.html?path=tests/test_file_limits.py#L1"
        }
      ],
      "kind": "store",
      "dependencies": [],
      "status": "experimental"
    },
    {
      "id": "browser",
      "name": "Dashboard browser UI",
      "group": "dashboard",
      "purpose": "Display recorded metrics, colors and cubes. Submit bounded inspection inputs to the local server.",
      "inputs": "Saved result JSON and inspector responses",
      "outputs": "Charts, tables and downloadable JSON",
      "side_effects": "Fetches local API endpoints. Export creates browser Blob downloads.",
      "ownership": "Browser memory owns UI and cube state",
      "trust_boundary": "Browser inputs are untrusted to the local server. Recorded metrics remain distinct from live inspection.",
      "failures": "Missing results or rejected inspection appears as an error. Browser does not run native hue operators.",
      "sources": [
        {
          "path": "dashboard/app.js",
          "contains": "async function encodeInput",
          "line": 322,
          "end_line": 322,
          "sha256": "59096088da2640564034b45108d19cd70477c3affa02954b30a03773fd7bb9a7",
          "url": "source.html?path=dashboard/app.js#L322"
        },
        {
          "path": "dashboard/cube.js",
          "contains": "fetch(\"/api/cube\"",
          "line": 83,
          "end_line": 83,
          "sha256": "22dae4dcecbf2e72e4c9140596950b8436910797a82c4646b279f21ea3804355",
          "url": "source.html?path=dashboard/cube.js#L83"
        }
      ],
      "tests": [
        {
          "path": "tests/test_dashboard.py",
          "symbol": "test_http_serves_measured_results_and_static_assets",
          "line": 70,
          "end_line": 83,
          "sha256": "76608e25cb244b7c5f935859d5a7cf8752729bffa9962a3a6c22deb8572bf966",
          "url": "source.html?path=tests/test_dashboard.py#L70"
        }
      ],
      "kind": "component",
      "dependencies": [],
      "status": "implemented"
    },
    {
      "id": "http",
      "name": "Local HTTP boundary",
      "group": "dashboard",
      "purpose": "Serve allowlisted static files, recorded results and bounded POST inspections. Validate exact local Host and Origin authorities.",
      "inputs": "Local HTTP requests, JSON bodies and artifact paths",
      "outputs": "HTTP results or explicit error status",
      "side_effects": "Reads results and permitted artifacts. Runs codec inspections on POST.",
      "ownership": "ThreadingHTTPServer owns connections and handlers",
      "trust_boundary": "Host, Origin, body size and resolved artifact paths gate local access. There is no account authentication.",
      "failures": "Foreign Host returns 421, cross-origin POST 403, wrong type 415, oversized body 413. Missing records return 404.",
      "sources": [
        {
          "path": "dashboard/server.py",
          "symbol": "DashboardHandler",
          "line": 129,
          "end_line": 280,
          "sha256": "c9ee7664b55deb67b03c6ed051ff3f8eeeca4576befce7cdd103ee871ccf2f2d",
          "url": "source.html?path=dashboard/server.py#L129"
        },
        {
          "path": "dashboard/server.py",
          "symbol": "DashboardServer",
          "line": 106,
          "end_line": 126,
          "sha256": "c9ee7664b55deb67b03c6ed051ff3f8eeeca4576befce7cdd103ee871ccf2f2d",
          "url": "source.html?path=dashboard/server.py#L106"
        }
      ],
      "tests": [
        {
          "path": "tests/test_dashboard.py",
          "symbol": "test_foreign_host_cannot_read_or_compute_even_with_matching_origin",
          "line": 120,
          "end_line": 126,
          "sha256": "76608e25cb244b7c5f935859d5a7cf8752729bffa9962a3a6c22deb8572bf966",
          "url": "source.html?path=tests/test_dashboard.py#L120"
        }
      ],
      "kind": "component",
      "dependencies": [
        "inspection"
      ],
      "status": "implemented"
    },
    {
      "id": "inspection",
      "name": "Legacy tensor and cube inspector",
      "group": "dashboard",
      "purpose": "Validate finite input values and cube state, then call the actual legacy codecs and cube APIs.",
      "inputs": "Values, shapes, codec name or cube operation and state",
      "outputs": "Codes, decoded values, geometry and measured error",
      "side_effects": "Allocates bounded PyTorch state. It can read selected saved samples.",
      "ownership": "Request creates temporary Python tensors. Cube state returns to the browser.",
      "trust_boundary": "Request JSON crosses type, shape and allocation limits",
      "failures": "Invalid values, shape, codec, cell identity or geometry returns a rejection. Missing PyTorch returns 503.",
      "sources": [
        {
          "path": "dashboard/server.py",
          "symbol": "inspect_tensor",
          "line": 25,
          "end_line": 103,
          "sha256": "c9ee7664b55deb67b03c6ed051ff3f8eeeca4576befce7cdd103ee871ccf2f2d",
          "url": "source.html?path=dashboard/server.py#L25"
        },
        {
          "path": "dashboard/cube_api.py",
          "symbol": "inspect_cube",
          "line": 150,
          "end_line": 212,
          "sha256": "30a6099e2f2726c498f300d213e0ad55c8ed8f2bf68b49597cfe247fbe58f472",
          "url": "source.html?path=dashboard/cube_api.py#L150"
        },
        {
          "path": "dashboard/cube_api.py",
          "symbol": "_restore",
          "line": 25,
          "end_line": 67,
          "sha256": "30a6099e2f2726c498f300d213e0ad55c8ed8f2bf68b49597cfe247fbe58f472",
          "url": "source.html?path=dashboard/cube_api.py#L25"
        }
      ],
      "tests": [
        {
          "path": "tests/test_dashboard.py",
          "symbol": "test_inspector_rejects_invalid_or_unbounded_requests",
          "line": 65,
          "end_line": 67,
          "sha256": "76608e25cb244b7c5f935859d5a7cf8752729bffa9962a3a6c22deb8572bf966",
          "url": "source.html?path=tests/test_dashboard.py#L65"
        },
        {
          "path": "tests/test_cube_api.py",
          "line": 1,
          "end_line": 282,
          "sha256": "2a1827ad175871cd8e2e4b287513b6a1a3203c6255fadf98f87e58286dcbd184",
          "url": "source.html?path=tests/test_cube_api.py#L1"
        }
      ],
      "kind": "component",
      "dependencies": [
        "codecs",
        "cubes"
      ],
      "status": "implemented"
    },
    {
      "id": "datasets",
      "name": "Dataset loaders",
      "group": "research",
      "purpose": "Select seeded official image subsets, sampled language windows and separately labeled analytic scalar data.",
      "inputs": "Task, seed, counts, cache path and device",
      "outputs": "Train and validation tensors with source URLs and hashes",
      "side_effects": "May download MNIST, CIFAR-10 and Tiny Shakespeare into the requested cache.",
      "ownership": "Experiment runner owns downloaded cache and selected tensors",
      "trust_boundary": "External dataset bytes enter training. Hashes record identity, not a fixed upstream authenticity guarantee.",
      "failures": "Download, decode, missing cache or invalid sample count fails. No synthetic substitution under a real dataset name.",
      "sources": [
        {
          "path": "experiments/datasets.py",
          "symbol": "image_data",
          "line": 42,
          "end_line": 63,
          "sha256": "d79df91b2ebad6bb51c02d0f96d69d2f0e674bfbecb8027441306910eb37e9f2",
          "url": "source.html?path=experiments/datasets.py#L42"
        },
        {
          "path": "experiments/datasets.py",
          "symbol": "language_data",
          "line": 74,
          "end_line": 96,
          "sha256": "d79df91b2ebad6bb51c02d0f96d69d2f0e674bfbecb8027441306910eb37e9f2",
          "url": "source.html?path=experiments/datasets.py#L74"
        },
        {
          "path": "experiments/datasets.py",
          "symbol": "scalar_data",
          "line": 66,
          "end_line": 71,
          "sha256": "d79df91b2ebad6bb51c02d0f96d69d2f0e674bfbecb8027441306910eb37e9f2",
          "url": "source.html?path=experiments/datasets.py#L66"
        }
      ],
      "tests": [
        {
          "path": "tests/test_datasets.py",
          "line": 1,
          "end_line": 37,
          "sha256": "aeb1c8fbe0a16a4f348bcbb28923e0bce3b958b42b953c57c70ca5eceeaa1331",
          "url": "source.html?path=tests/test_datasets.py#L1"
        }
      ],
      "kind": "integration",
      "dependencies": [],
      "status": "experimental"
    },
    {
      "id": "training",
      "name": "Research runners",
      "group": "research",
      "purpose": "Run seeded training, attention, ablations and cube comparisons with explicit precision and equal-budget controls.",
      "inputs": "Configuration, datasets, seeds, representation and device",
      "outputs": "Measurements, curves, samples and model checkpoints",
      "side_effects": "Trains models and writes artifacts and JSON reports.",
      "ownership": "Runner owns temporary models and run directory",
      "trust_boundary": "Execution metadata must distinguish floating training, integer controls and native operators",
      "failures": "Unavailable device, data, dependency or failed experiment must remain visible in evidence.",
      "sources": [
        {
          "path": "experiments/training.py",
          "symbol": "train_one",
          "line": 129,
          "end_line": 225,
          "sha256": "936c0791667a84e9b1aed08966c5ba090454ad87c5bf31a082ac2859553bb614",
          "url": "source.html?path=experiments/training.py#L129"
        },
        {
          "path": "scripts/run_benchmarks.py",
          "symbol": "main",
          "line": 24,
          "end_line": 137,
          "sha256": "3c666add71610352bebffe13b09689f3977901b443db13959b0c4e3478fbd236",
          "url": "source.html?path=scripts/run_benchmarks.py#L24"
        },
        {
          "path": "scripts/run_cube_experiments.py",
          "symbol": "main",
          "line": 16,
          "end_line": 93,
          "sha256": "d65dccf54be8aaf4a2af3dc2ae73068454edc9dc9bfa428dfcfdc49fdd90ec37",
          "url": "source.html?path=scripts/run_cube_experiments.py#L16"
        }
      ],
      "tests": [
        {
          "path": "tests/test_training.py",
          "line": 1,
          "end_line": 137,
          "sha256": "e159375ee9c0605212a1ab5043ed8497607ab8176ce81f588017ac426777c50b",
          "url": "source.html?path=tests/test_training.py#L1"
        },
        {
          "path": "tests/test_cube_experiments.py",
          "line": 1,
          "end_line": 180,
          "sha256": "39309ef4d9342d1fa9aa43de1cd31bb15af8de3ab42e92c55de99286f42b11bb",
          "url": "source.html?path=tests/test_cube_experiments.py#L1"
        }
      ],
      "kind": "component",
      "dependencies": [
        "cubes",
        "legacy-storage",
        "nn"
      ],
      "status": "experimental"
    },
    {
      "id": "provenance",
      "name": "Provenance and result checks",
      "group": "research",
      "purpose": "Bind configuration, executable source and artifact hashes to measured records. Check coverage, precision and artifact paths.",
      "inputs": "Run records, source files, config and artifact paths",
      "outputs": "Verified identities or gate failures",
      "side_effects": "Reads and hashes source, artifacts and metadata.",
      "ownership": "Report owns a manifest snapshot. Checker compares it to current files.",
      "trust_boundary": "Recorded claims cross into acceptance only after independent checks",
      "failures": "Source drift, missing artifacts, unsafe paths, false coverage or execution claims fail.",
      "sources": [
        {
          "path": "experiments/provenance.py",
          "symbol": "source_manifest",
          "line": 32,
          "end_line": 44,
          "sha256": "abe3a22747f65c04133ed253425b2fe1b9811ddf9f2a0fe804b72fd143d6eb5f",
          "url": "source.html?path=experiments/provenance.py#L32"
        },
        {
          "path": "experiments/provenance.py",
          "symbol": "verify_provenance",
          "line": 115,
          "end_line": 140,
          "sha256": "abe3a22747f65c04133ed253425b2fe1b9811ddf9f2a0fe804b72fd143d6eb5f",
          "url": "source.html?path=experiments/provenance.py#L115"
        },
        {
          "path": "experiments/verification.py",
          "symbol": "verify_precision",
          "line": 166,
          "end_line": 180,
          "sha256": "7ccc8e9bd22c7d98bb13c97fa52ca6f9dcdab99853c7d80a4a6411f898dff317",
          "url": "source.html?path=experiments/verification.py#L166"
        }
      ],
      "tests": [
        {
          "path": "tests/test_acceptance.py",
          "line": 1,
          "end_line": 463,
          "sha256": "82cfb263ea980b2db6c1b9d78eb1ec4ef536c5dcd78ac57200aadb5869c5121b",
          "url": "source.html?path=tests/test_acceptance.py#L1"
        },
        {
          "path": "tests/test_release_research.py",
          "line": 1,
          "end_line": 196,
          "sha256": "afa406fab27e92708eb1f5912a3e06ef942d3f27ef7b0ab540d190770c71108f",
          "url": "source.html?path=tests/test_release_research.py#L1"
        }
      ],
      "kind": "component",
      "dependencies": [
        "acceptance",
        "legacy-gates"
      ],
      "status": "experimental"
    },
    {
      "id": "results",
      "name": "Local results and artifacts",
      "group": "research",
      "purpose": "Store measured JSON, saved models, native tensor files and receipts in results and artifacts directories.",
      "inputs": "Runner output, checkpoints and evidence",
      "outputs": "Files read by checkers and dashboard",
      "side_effects": "Persistent local writes. No cloud database or remote synchronization.",
      "ownership": "Workspace owner controls local files",
      "trust_boundary": "Results are mutable local files. Checksums alone do not authenticate the author.",
      "failures": "Absent, stale, corrupt or incompatible records cannot establish current acceptance.",
      "sources": [
        {
          "path": "experiments/provenance.py",
          "symbol": "artifact_manifest",
          "line": 69,
          "end_line": 85,
          "sha256": "abe3a22747f65c04133ed253425b2fe1b9811ddf9f2a0fe804b72fd143d6eb5f",
          "url": "source.html?path=experiments/provenance.py#L69"
        },
        {
          "path": "dashboard/server.py",
          "contains": "if request.path in (\"/api/results\", \"/api/cube-results\")",
          "line": 222,
          "end_line": 222,
          "sha256": "c9ee7664b55deb67b03c6ed051ff3f8eeeca4576befce7cdd103ee871ccf2f2d",
          "url": "source.html?path=dashboard/server.py#L222"
        }
      ],
      "tests": [
        {
          "path": "tests/test_dashboard.py",
          "symbol": "test_invalid_results_cannot_be_presented_as_measurements",
          "line": 95,
          "end_line": 100,
          "sha256": "76608e25cb244b7c5f935859d5a7cf8752729bffa9962a3a6c22deb8572bf966",
          "url": "source.html?path=tests/test_dashboard.py#L95"
        }
      ],
      "kind": "store",
      "dependencies": [],
      "status": "experimental"
    },
    {
      "id": "reporting",
      "name": "Research report writer",
      "group": "research",
      "purpose": "Compute paired comparisons and write conclusions based on recorded measurements.",
      "inputs": "Run records and comparison pairs",
      "outputs": "RESULTS prose and comparison metadata",
      "side_effects": "Writes report documents.",
      "ownership": "Run records remain the measurement source",
      "trust_boundary": "A report does not add hardware, energy, novelty or generalization evidence",
      "failures": "Missing or unsupported metrics limit conclusions. Native CPU results do not establish a speed benefit.",
      "sources": [
        {
          "path": "experiments/reporting.py",
          "symbol": "paired_comparisons",
          "line": 9,
          "end_line": 23,
          "sha256": "0b4a0ac3f82a149dc3258b8c000b8530115b3ce4982bb650163b42e6d71027ec",
          "url": "source.html?path=experiments/reporting.py#L9"
        },
        {
          "path": "experiments/reporting.py",
          "symbol": "conclusions",
          "line": 26,
          "end_line": 50,
          "sha256": "0b4a0ac3f82a149dc3258b8c000b8530115b3ce4982bb650163b42e6d71027ec",
          "url": "source.html?path=experiments/reporting.py#L26"
        }
      ],
      "tests": [
        {
          "path": "tests/test_release_research.py",
          "line": 1,
          "end_line": 196,
          "sha256": "afa406fab27e92708eb1f5912a3e06ef942d3f27ef7b0ab540d190770c71108f",
          "url": "source.html?path=tests/test_release_research.py#L1"
        }
      ],
      "kind": "component",
      "dependencies": [],
      "status": "experimental"
    },
    {
      "id": "build",
      "name": "Native builder and wheel",
      "group": "release",
      "purpose": "Check Rust attribution hashes and compiler identity, build in offline mode and assemble a wheel containing the exact binding, C header and runtime notices.",
      "inputs": "Rust sources, binding, attribution manifest, matching compiler and target platform",
      "outputs": "Library, CLI, platform wheel and hashes",
      "side_effects": "Runs Cargo. Linux cross-build runs a pinned container. Writes platform build, native library and wheel directories.",
      "ownership": "Builder owns build outputs. Installer owns installed files.",
      "trust_boundary": "Build toolchain and platform binary cross the packaging boundary",
      "failures": "Missing Cargo, container, unsupported target or build failure aborts. Build output does not prove runtime correctness.",
      "sources": [
        {
          "path": "scripts/build_native_hue.py",
          "symbol": "rust_notices",
          "line": 69,
          "end_line": 78,
          "sha256": "354fd81cdef0aa748b7b3eea4adc8b630c3f0c939468c7f67b27ffe510ca9102",
          "url": "source.html?path=scripts/build_native_hue.py#L69"
        },
        {
          "path": "scripts/build_native_hue.py",
          "symbol": "make_wheel",
          "line": 86,
          "end_line": 126,
          "sha256": "354fd81cdef0aa748b7b3eea4adc8b630c3f0c939468c7f67b27ffe510ca9102",
          "url": "source.html?path=scripts/build_native_hue.py#L86"
        },
        {
          "path": "scripts/build_native_hue.py",
          "symbol": "main",
          "line": 129,
          "end_line": 197,
          "sha256": "354fd81cdef0aa748b7b3eea4adc8b630c3f0c939468c7f67b27ffe510ca9102",
          "url": "source.html?path=scripts/build_native_hue.py#L129"
        }
      ],
      "tests": [
        {
          "path": "tests/test_native_release.py",
          "line": 1,
          "end_line": 110,
          "sha256": "a9361b1648082c4294124683974016a2d2602635c38baac6a8d56047f51797d9",
          "url": "source.html?path=tests/test_native_release.py#L1"
        }
      ],
      "kind": "component",
      "dependencies": [],
      "status": "implemented"
    },
    {
      "id": "acceptance",
      "name": "Native release gates",
      "group": "release",
      "purpose": "Check fresh sources, host tests, Rust builds, independent integer oracle, CLI proof and isolated wheel environments.",
      "inputs": "Current source, binaries, receipts and installed-wheel test records",
      "outputs": "Native acceptance report with passed or failed gates",
      "side_effects": "Reads artifacts and writes verification JSON. Package runner creates isolated environments.",
      "ownership": "Checker owns the report. Each receipt identifies its execution layer.",
      "trust_boundary": "Historical evidence cannot claim acceptance for changed sources or binaries",
      "failures": "Source mismatch, tampered oracle, missing operation, invalid package matrix or failed command fails acceptance.",
      "sources": [
        {
          "path": "scripts/verify_native_hue_release.py",
          "symbol": "verify",
          "line": 122,
          "end_line": 269,
          "sha256": "6117f5fdcdcb1019e7c92b9f4b8d4183498ff355da43e2a2477418c903a50b4e",
          "url": "source.html?path=scripts/verify_native_hue_release.py#L122"
        },
        {
          "path": "scripts/verify_native_hue.py",
          "symbol": "main",
          "line": 416,
          "end_line": 435,
          "sha256": "dabad2682a51660bb68a3fffa76243572a5960f4ed79c5b2b507c3a8eaf5a648",
          "url": "source.html?path=scripts/verify_native_hue.py#L416"
        },
        {
          "path": "scripts/run_native_hue_package_checks.py",
          "symbol": "main",
          "line": 29,
          "end_line": 76,
          "sha256": "bd1ecf8da6fa03088ab99b254757157c2fb23482a55bc73e26fd5ae5490ff81d",
          "url": "source.html?path=scripts/run_native_hue_package_checks.py#L29"
        }
      ],
      "tests": [
        {
          "path": "tests/test_native_release.py",
          "line": 1,
          "end_line": 110,
          "sha256": "a9361b1648082c4294124683974016a2d2602635c38baac6a8d56047f51797d9",
          "url": "source.html?path=tests/test_native_release.py#L1"
        },
        {
          "path": "tests/test_native_hue.py",
          "symbol": "test_rehashed_integer_control_tampering_and_false_execution_claims_are_rejected",
          "line": 370,
          "end_line": 384,
          "sha256": "58d60cc357bf4734cc83a7a91d39b161919643229bc465433eeef0748e1ed8dd",
          "url": "source.html?path=tests/test_native_hue.py#L370"
        }
      ],
      "kind": "component",
      "dependencies": [],
      "status": "implemented"
    },
    {
      "id": "legacy-gates",
      "name": "Legacy research gates",
      "group": "release",
      "purpose": "Select the separate legacy acceptance path and check research coverage, artifacts and dependency matrix.",
      "inputs": "Legacy result and test records",
      "outputs": "Separate legacy acceptance JSON",
      "side_effects": "Reads records and writes a report.",
      "ownership": "Legacy checker owns a report separate from native release",
      "trust_boundary": "Passing research gates does not establish native GPU, autograd or neural training",
      "failures": "Missing tests, stale source, incomplete matrix or inconsistent metrics fails.",
      "sources": [
        {
          "path": "scripts/verify.py",
          "symbol": "verify",
          "line": 18,
          "end_line": 100,
          "sha256": "edb0c32960052ffa22d32d029aaf2fcac17ecf290e31efe34aa96c1a8d0cd30e",
          "url": "source.html?path=scripts/verify.py#L18"
        }
      ],
      "tests": [
        {
          "path": "tests/test_acceptance.py",
          "line": 1,
          "end_line": 463,
          "sha256": "82cfb263ea980b2db6c1b9d78eb1ec4ef536c5dcd78ac57200aadb5869c5121b",
          "url": "source.html?path=tests/test_acceptance.py#L1"
        },
        {
          "path": "tests/test_release_matrix.py",
          "line": 1,
          "end_line": 31,
          "sha256": "54edcc832000ccda014dbfe0ab0a349cfef165a6a127a4a6aeeac41fa1a4a014",
          "url": "source.html?path=tests/test_release_matrix.py#L1"
        }
      ],
      "kind": "component",
      "dependencies": [],
      "status": "implemented"
    }
  ],
  "edges": [
    {
      "id": "package-python",
      "source": "package",
      "target": "python",
      "type": "call",
      "label": "Native imports and aliases",
      "evidence": [
        {
          "path": "chromatensor/__init__.py",
          "contains": "from .hue import",
          "line": 4,
          "end_line": 4,
          "sha256": "5a0d9049c45cabb47f36f66a6800cd714d3b0ff8463f3a275f1eb9948be432c7",
          "url": "source.html?path=chromatensor/__init__.py#L4"
        }
      ],
      "boundary": false,
      "flows": [
        "native"
      ]
    },
    {
      "id": "python-ffi",
      "source": "python",
      "target": "ffi",
      "type": "call",
      "label": "ctypes with ordered handle locks",
      "evidence": [
        {
          "path": "chromatensor/hue.py",
          "symbol": "_owned",
          "line": 138,
          "end_line": 150,
          "sha256": "9781800dbbce9893283390ee3b409027f95d48a5fcef01e48864ea5cb3126c97",
          "url": "source.html?path=chromatensor/hue.py#L138"
        },
        {
          "path": "chromatensor/hue.py",
          "symbol": "_library",
          "line": 46,
          "end_line": 78,
          "sha256": "9781800dbbce9893283390ee3b409027f95d48a5fcef01e48864ea5cb3126c97",
          "url": "source.html?path=chromatensor/hue.py#L46"
        }
      ],
      "boundary": true,
      "flows": [
        "native",
        "save-load"
      ]
    },
    {
      "id": "header-ffi",
      "source": "header",
      "target": "ffi",
      "type": "contract",
      "label": "Declares C ABI and caller obligations",
      "evidence": [
        {
          "path": "native/hue_tensor/include/hue_tensor.h",
          "contains": "void ht_free",
          "line": 23,
          "end_line": 23,
          "sha256": "81eaefa1bc501b8de9f96875a5a1723799db2532e8ba7b07bf7d6cd0c63a3f6c",
          "url": "source.html?path=native/hue_tensor/include/hue_tensor.h#L23"
        }
      ],
      "boundary": true,
      "flows": []
    },
    {
      "id": "ffi-tensor",
      "source": "ffi",
      "target": "tensor",
      "type": "ownership",
      "label": "Allocate opaque handle, free once",
      "evidence": [
        {
          "path": "native/hue_tensor/src/ffi.rs",
          "symbol": "ht_new_rgbd",
          "line": 153,
          "end_line": 153,
          "sha256": "cd4e88e8cdd9be41c9196d3f8e22baf82c376398290ddf0c7f0b1ecf2144f920",
          "url": "source.html?path=native/hue_tensor/src/ffi.rs#L153"
        },
        {
          "path": "native/hue_tensor/src/ffi.rs",
          "symbol": "ht_free",
          "line": 222,
          "end_line": 222,
          "sha256": "cd4e88e8cdd9be41c9196d3f8e22baf82c376398290ddf0c7f0b1ecf2144f920",
          "url": "source.html?path=native/hue_tensor/src/ffi.rs#L222"
        }
      ],
      "boundary": true,
      "flows": [
        "native"
      ]
    },
    {
      "id": "tokens-tensor",
      "source": "tokens",
      "target": "tensor",
      "type": "data",
      "label": "Validated literal RGBD tokens",
      "evidence": [
        {
          "path": "native/hue_tensor/src/lib.rs",
          "symbol": "from_rgbd",
          "line": 216,
          "end_line": 216,
          "sha256": "ab8c42dbcd5750c20f16534c6adf97a3a11b501e898b0583a1fd897abc6c0ccb",
          "url": "source.html?path=native/hue_tensor/src/lib.rs#L216"
        }
      ],
      "boundary": false,
      "flows": [
        "native"
      ]
    },
    {
      "id": "tensor-operators",
      "source": "tensor",
      "target": "operators",
      "type": "data",
      "label": "Borrow token and shape inputs",
      "evidence": [
        {
          "path": "native/hue_tensor/src/lib.rs",
          "symbol": "matmul",
          "line": 328,
          "end_line": 328,
          "sha256": "ab8c42dbcd5750c20f16534c6adf97a3a11b501e898b0583a1fd897abc6c0ccb",
          "url": "source.html?path=native/hue_tensor/src/lib.rs#L328"
        }
      ],
      "boundary": false,
      "flows": [
        "native"
      ]
    },
    {
      "id": "operators-tokens",
      "source": "operators",
      "target": "tokens",
      "type": "call",
      "label": "Encode integer results into tokens",
      "evidence": [
        {
          "path": "native/hue_tensor/src/lib.rs",
          "symbol": "Tensor::binary",
          "line": 267,
          "end_line": 267,
          "sha256": "ab8c42dbcd5750c20f16534c6adf97a3a11b501e898b0583a1fd897abc6c0ccb",
          "url": "source.html?path=native/hue_tensor/src/lib.rs#L267"
        }
      ],
      "boundary": false,
      "flows": [
        "native"
      ]
    },
    {
      "id": "operators-tensor",
      "source": "operators",
      "target": "tensor",
      "type": "ownership",
      "label": "Return a new owned output",
      "evidence": [
        {
          "path": "native/hue_tensor/src/lib.rs",
          "symbol": "reshape",
          "line": 399,
          "end_line": 399,
          "sha256": "ab8c42dbcd5750c20f16534c6adf97a3a11b501e898b0583a1fd897abc6c0ccb",
          "url": "source.html?path=native/hue_tensor/src/lib.rs#L399"
        }
      ],
      "boundary": false,
      "flows": [
        "native"
      ]
    },
    {
      "id": "ffi-storage",
      "source": "ffi",
      "target": "storage",
      "type": "call",
      "label": "ht_save and ht_load",
      "evidence": [
        {
          "path": "native/hue_tensor/src/ffi.rs",
          "symbol": "ht_save",
          "line": 501,
          "end_line": 501,
          "sha256": "cd4e88e8cdd9be41c9196d3f8e22baf82c376398290ddf0c7f0b1ecf2144f920",
          "url": "source.html?path=native/hue_tensor/src/ffi.rs#L501"
        },
        {
          "path": "native/hue_tensor/src/ffi.rs",
          "symbol": "ht_load",
          "line": 511,
          "end_line": 511,
          "sha256": "cd4e88e8cdd9be41c9196d3f8e22baf82c376398290ddf0c7f0b1ecf2144f920",
          "url": "source.html?path=native/hue_tensor/src/ffi.rs#L511"
        }
      ],
      "boundary": true,
      "flows": [
        "save-load"
      ]
    },
    {
      "id": "storage-tensor",
      "source": "storage",
      "target": "tensor",
      "type": "read",
      "label": "Read and validate saved RGBD and shape",
      "evidence": [
        {
          "path": "native/hue_tensor/src/storage.rs",
          "symbol": "load_tensor",
          "line": 167,
          "end_line": 167,
          "sha256": "204dcf09f6f17e6f64b806aa157f9b1af207a0b3cd0c3ec78d6661adca11c137",
          "url": "source.html?path=native/hue_tensor/src/storage.rs#L167"
        }
      ],
      "boundary": true,
      "flows": [
        "save-load"
      ]
    },
    {
      "id": "tensor-storage",
      "source": "tensor",
      "target": "storage",
      "type": "write",
      "label": "Write literal tokens, shape and checksum",
      "evidence": [
        {
          "path": "native/hue_tensor/src/storage.rs",
          "symbol": "save_tensor",
          "line": 103,
          "end_line": 103,
          "sha256": "204dcf09f6f17e6f64b806aa157f9b1af207a0b3cd0c3ec78d6661adca11c137",
          "url": "source.html?path=native/hue_tensor/src/storage.rs#L103"
        }
      ],
      "boundary": true,
      "flows": [
        "save-load"
      ]
    },
    {
      "id": "cli-tensor",
      "source": "cli",
      "target": "tensor",
      "type": "call",
      "label": "Direct Rust constructors",
      "evidence": [
        {
          "path": "native/hue_tensor/src/bin/hue_tensor.rs",
          "symbol": "run",
          "line": 236,
          "end_line": 236,
          "sha256": "46438f9dc38d96ab1703875a883c8993fccdf77430851bda22746a4c03d3650c",
          "url": "source.html?path=native/hue_tensor/src/bin/hue_tensor.rs#L236"
        }
      ],
      "boundary": true,
      "flows": [
        "cli"
      ]
    },
    {
      "id": "cli-operators",
      "source": "cli",
      "target": "operators",
      "type": "call",
      "label": "Direct Rust arithmetic",
      "evidence": [
        {
          "path": "native/hue_tensor/src/bin/hue_tensor.rs",
          "symbol": "run",
          "line": 236,
          "end_line": 236,
          "sha256": "46438f9dc38d96ab1703875a883c8993fccdf77430851bda22746a4c03d3650c",
          "url": "source.html?path=native/hue_tensor/src/bin/hue_tensor.rs#L236"
        }
      ],
      "boundary": false,
      "flows": [
        "cli"
      ]
    },
    {
      "id": "cli-storage",
      "source": "cli",
      "target": "storage",
      "type": "call",
      "label": "Bounded file read and save",
      "evidence": [
        {
          "path": "native/hue_tensor/src/bin/hue_tensor.rs",
          "symbol": "load",
          "line": 150,
          "end_line": 150,
          "sha256": "46438f9dc38d96ab1703875a883c8993fccdf77430851bda22746a4c03d3650c",
          "url": "source.html?path=native/hue_tensor/src/bin/hue_tensor.rs#L150"
        },
        {
          "path": "native/hue_tensor/src/bin/hue_tensor.rs",
          "symbol": "write_and_print",
          "line": 144,
          "end_line": 144,
          "sha256": "46438f9dc38d96ab1703875a883c8993fccdf77430851bda22746a4c03d3650c",
          "url": "source.html?path=native/hue_tensor/src/bin/hue_tensor.rs#L144"
        }
      ],
      "boundary": true,
      "flows": [
        "cli"
      ]
    },
    {
      "id": "package-legacy",
      "source": "package",
      "target": "legacy-tensor",
      "type": "call",
      "label": "Requested legacy name loads its module",
      "evidence": [
        {
          "path": "chromatensor/__init__.py",
          "symbol": "__getattr__",
          "line": 19,
          "end_line": 24,
          "sha256": "5a0d9049c45cabb47f36f66a6800cd714d3b0ff8463f3a275f1eb9948be432c7",
          "url": "source.html?path=chromatensor/__init__.py#L19"
        }
      ],
      "boundary": true,
      "flows": []
    },
    {
      "id": "legacy-codecs",
      "source": "legacy-tensor",
      "target": "codecs",
      "type": "call",
      "label": "Encode and decode scalar values",
      "evidence": [
        {
          "path": "chromatensor/tensor.py",
          "symbol": "ChromaTensor",
          "line": 14,
          "end_line": 221,
          "sha256": "7daff1f6685c23d2713a2879f18480d1097d878af50c82487628486ac8f72ed8",
          "url": "source.html?path=chromatensor/tensor.py#L14"
        }
      ],
      "boundary": false,
      "flows": []
    },
    {
      "id": "legacy-ops-tensor",
      "source": "legacy-ops",
      "target": "legacy-tensor",
      "type": "call",
      "label": "Decode, PyTorch operation, re-encode",
      "evidence": [
        {
          "path": "chromatensor/operators/core.py",
          "symbol": "_decoded",
          "line": 10,
          "end_line": 14,
          "sha256": "5db3a66e45c15f764845c71d92053f1d60f0d6bd7c6632b0d3fcabf6d47c35a7",
          "url": "source.html?path=chromatensor/operators/core.py#L10"
        }
      ],
      "boundary": false,
      "flows": [
        "training"
      ]
    },
    {
      "id": "nn-codecs",
      "source": "nn",
      "target": "codecs",
      "type": "data",
      "label": "Quantizer encoded state and color codes",
      "evidence": [
        {
          "path": "chromatensor/autograd/quantization.py",
          "symbol": "ChromaQuantizer",
          "line": 126,
          "end_line": 215,
          "sha256": "9524cd0af99950a2852a84cadbd29b8d374a089dce01f793cf903da9e72fe393",
          "url": "source.html?path=chromatensor/autograd/quantization.py#L126"
        }
      ],
      "boundary": false,
      "flows": [
        "training"
      ]
    },
    {
      "id": "cubes-legacy",
      "source": "cubes",
      "target": "legacy-tensor",
      "type": "call",
      "label": "Encoded ChromaTensor cell grid",
      "evidence": [
        {
          "path": "chromatensor/cube.py",
          "symbol": "ChromaCube",
          "line": 25,
          "end_line": 278,
          "sha256": "743086b6121e60630681e4958a65f9bd2195e71ef04520ef43ad8fc2dbd24451",
          "url": "source.html?path=chromatensor/cube.py#L25"
        }
      ],
      "boundary": false,
      "flows": [
        "inspection"
      ]
    },
    {
      "id": "legacy-storage-tensor",
      "source": "legacy-storage",
      "target": "legacy-tensor",
      "type": "data",
      "label": "Legacy codes and codec configuration",
      "evidence": [
        {
          "path": "chromatensor/serialization/format.py",
          "symbol": "load_ctensor",
          "line": 231,
          "end_line": 236,
          "sha256": "6b2694152b7a2d1face0eb6db891ac6effea9a0d06287657b63be97e9936e8b1",
          "url": "source.html?path=chromatensor/serialization/format.py#L231"
        }
      ],
      "boundary": false,
      "flows": []
    },
    {
      "id": "browser-http",
      "source": "browser",
      "target": "http",
      "type": "http",
      "label": "Local GET results, POST inspections",
      "evidence": [
        {
          "path": "dashboard/app.js",
          "contains": "fetch(\"/api/results\"",
          "line": 334,
          "end_line": 334,
          "sha256": "59096088da2640564034b45108d19cd70477c3affa02954b30a03773fd7bb9a7",
          "url": "source.html?path=dashboard/app.js#L334"
        },
        {
          "path": "dashboard/cube.js",
          "contains": "fetch(\"/api/cube\"",
          "line": 83,
          "end_line": 83,
          "sha256": "22dae4dcecbf2e72e4c9140596950b8436910797a82c4646b279f21ea3804355",
          "url": "source.html?path=dashboard/cube.js#L83"
        }
      ],
      "boundary": true,
      "flows": [
        "inspection"
      ]
    },
    {
      "id": "http-browser",
      "source": "http",
      "target": "browser",
      "type": "http",
      "label": "JSON responses and static assets",
      "evidence": [
        {
          "path": "dashboard/server.py",
          "symbol": "DashboardHandler",
          "line": 129,
          "end_line": 280,
          "sha256": "c9ee7664b55deb67b03c6ed051ff3f8eeeca4576befce7cdd103ee871ccf2f2d",
          "url": "source.html?path=dashboard/server.py#L129"
        }
      ],
      "boundary": true,
      "flows": [
        "inspection"
      ]
    },
    {
      "id": "http-inspection",
      "source": "http",
      "target": "inspection",
      "type": "call",
      "label": "Validated JSON request",
      "evidence": [
        {
          "path": "dashboard/server.py",
          "contains": "result = inspect_tensor(payload)",
          "line": 273,
          "end_line": 273,
          "sha256": "c9ee7664b55deb67b03c6ed051ff3f8eeeca4576befce7cdd103ee871ccf2f2d",
          "url": "source.html?path=dashboard/server.py#L273"
        }
      ],
      "boundary": true,
      "flows": [
        "inspection"
      ]
    },
    {
      "id": "inspection-codecs",
      "source": "inspection",
      "target": "codecs",
      "type": "call",
      "label": "Actual legacy codec encode and decode",
      "evidence": [
        {
          "path": "dashboard/server.py",
          "symbol": "inspect_tensor",
          "line": 25,
          "end_line": 103,
          "sha256": "c9ee7664b55deb67b03c6ed051ff3f8eeeca4576befce7cdd103ee871ccf2f2d",
          "url": "source.html?path=dashboard/server.py#L25"
        }
      ],
      "boundary": false,
      "flows": [
        "inspection"
      ]
    },
    {
      "id": "inspection-cubes",
      "source": "inspection",
      "target": "cubes",
      "type": "call",
      "label": "Restore, twist, inspect and reconstruct",
      "evidence": [
        {
          "path": "dashboard/cube_api.py",
          "symbol": "inspect_cube",
          "line": 150,
          "end_line": 212,
          "sha256": "30a6099e2f2726c498f300d213e0ad55c8ed8f2bf68b49597cfe247fbe58f472",
          "url": "source.html?path=dashboard/cube_api.py#L150"
        }
      ],
      "boundary": false,
      "flows": [
        "inspection"
      ]
    },
    {
      "id": "results-http",
      "source": "results",
      "target": "http",
      "type": "read",
      "label": "Read allowlisted local results and artifacts",
      "evidence": [
        {
          "path": "dashboard/server.py",
          "contains": "if request.path == \"/api/artifact\"",
          "line": 234,
          "end_line": 234,
          "sha256": "c9ee7664b55deb67b03c6ed051ff3f8eeeca4576befce7cdd103ee871ccf2f2d",
          "url": "source.html?path=dashboard/server.py#L234"
        }
      ],
      "boundary": true,
      "flows": [
        "inspection"
      ]
    },
    {
      "id": "datasets-training",
      "source": "datasets",
      "target": "training",
      "type": "data",
      "label": "Seeded splits, counts and dataset hashes",
      "evidence": [
        {
          "path": "experiments/training.py",
          "contains": "from .datasets import",
          "line": 16,
          "end_line": 16,
          "sha256": "936c0791667a84e9b1aed08966c5ba090454ad87c5bf31a082ac2859553bb614",
          "url": "source.html?path=experiments/training.py#L16"
        }
      ],
      "boundary": true,
      "flows": [
        "training"
      ]
    },
    {
      "id": "training-nn",
      "source": "training",
      "target": "nn",
      "type": "call",
      "label": "Floating training and quantized layers",
      "evidence": [
        {
          "path": "experiments/training.py",
          "symbol": "train_one",
          "line": 129,
          "end_line": 225,
          "sha256": "936c0791667a84e9b1aed08966c5ba090454ad87c5bf31a082ac2859553bb614",
          "url": "source.html?path=experiments/training.py#L129"
        }
      ],
      "boundary": false,
      "flows": [
        "training"
      ]
    },
    {
      "id": "training-cubes",
      "source": "training",
      "target": "cubes",
      "type": "call",
      "label": "Cube geometry and learning experiments",
      "evidence": [
        {
          "path": "experiments/cube.py",
          "contains": "from chromatensor",
          "line": 10,
          "end_line": 10,
          "sha256": "745371c258c9fd10e50d753aea457e9d5305950d2e7fa7befb9583e7725860fe",
          "url": "source.html?path=experiments/cube.py#L10"
        }
      ],
      "boundary": false,
      "flows": []
    },
    {
      "id": "training-storage",
      "source": "training",
      "target": "legacy-storage",
      "type": "call",
      "label": "Measured checkpoint storage",
      "evidence": [
        {
          "path": "experiments/training.py",
          "symbol": "storage_metrics",
          "line": 91,
          "end_line": 126,
          "sha256": "936c0791667a84e9b1aed08966c5ba090454ad87c5bf31a082ac2859553bb614",
          "url": "source.html?path=experiments/training.py#L91"
        }
      ],
      "boundary": true,
      "flows": [
        "training"
      ]
    },
    {
      "id": "training-results",
      "source": "training",
      "target": "results",
      "type": "write",
      "label": "Write measurements and saved artifacts",
      "evidence": [
        {
          "path": "scripts/run_benchmarks.py",
          "symbol": "main",
          "line": 24,
          "end_line": 137,
          "sha256": "3c666add71610352bebffe13b09689f3977901b443db13959b0c4e3478fbd236",
          "url": "source.html?path=scripts/run_benchmarks.py#L24"
        }
      ],
      "boundary": true,
      "flows": [
        "training"
      ]
    },
    {
      "id": "results-provenance",
      "source": "results",
      "target": "provenance",
      "type": "read",
      "label": "Read, hash and validate saved run evidence",
      "evidence": [
        {
          "path": "experiments/provenance.py",
          "symbol": "verify_provenance",
          "line": 115,
          "end_line": 140,
          "sha256": "abe3a22747f65c04133ed253425b2fe1b9811ddf9f2a0fe804b72fd143d6eb5f",
          "url": "source.html?path=experiments/provenance.py#L115"
        }
      ],
      "boundary": true,
      "flows": [
        "release"
      ]
    },
    {
      "id": "results-reporting",
      "source": "results",
      "target": "reporting",
      "type": "data",
      "label": "Supply paired measured records",
      "evidence": [
        {
          "path": "experiments/reporting.py",
          "symbol": "conclusions",
          "line": 26,
          "end_line": 50,
          "sha256": "0b4a0ac3f82a149dc3258b8c000b8530115b3ce4982bb650163b42e6d71027ec",
          "url": "source.html?path=experiments/reporting.py#L26"
        }
      ],
      "boundary": false,
      "flows": []
    },
    {
      "id": "python-build",
      "source": "python",
      "target": "build",
      "type": "data",
      "label": "Exact binding copied to the native wheel",
      "evidence": [
        {
          "path": "scripts/build_native_hue.py",
          "symbol": "make_wheel",
          "line": 86,
          "end_line": 126,
          "sha256": "354fd81cdef0aa748b7b3eea4adc8b630c3f0c939468c7f67b27ffe510ca9102",
          "url": "source.html?path=scripts/build_native_hue.py#L86"
        }
      ],
      "boundary": true,
      "flows": [
        "release"
      ]
    },
    {
      "id": "header-build",
      "source": "header",
      "target": "build",
      "type": "data",
      "label": "C header copied to wheel",
      "evidence": [
        {
          "path": "scripts/build_native_hue.py",
          "symbol": "make_wheel",
          "line": 86,
          "end_line": 126,
          "sha256": "354fd81cdef0aa748b7b3eea4adc8b630c3f0c939468c7f67b27ffe510ca9102",
          "url": "source.html?path=scripts/build_native_hue.py#L86"
        }
      ],
      "boundary": false,
      "flows": [
        "release"
      ]
    },
    {
      "id": "build-acceptance",
      "source": "build",
      "target": "acceptance",
      "type": "data",
      "label": "Built binaries, wheel hashes and receipts",
      "evidence": [
        {
          "path": "scripts/verify_native_hue_release.py",
          "symbol": "verify",
          "line": 122,
          "end_line": 269,
          "sha256": "6117f5fdcdcb1019e7c92b9f4b8d4183498ff355da43e2a2477418c903a50b4e",
          "url": "source.html?path=scripts/verify_native_hue_release.py#L122"
        }
      ],
      "boundary": true,
      "flows": [
        "release"
      ]
    },
    {
      "id": "provenance-acceptance",
      "source": "provenance",
      "target": "acceptance",
      "type": "call",
      "label": "Current source and artifact identities",
      "evidence": [
        {
          "path": "scripts/verify_native_hue_release.py",
          "contains": "from experiments.provenance import",
          "line": 18,
          "end_line": 18,
          "sha256": "6117f5fdcdcb1019e7c92b9f4b8d4183498ff355da43e2a2477418c903a50b4e",
          "url": "source.html?path=scripts/verify_native_hue_release.py#L18"
        }
      ],
      "boundary": false,
      "flows": [
        "release"
      ]
    },
    {
      "id": "acceptance-results",
      "source": "acceptance",
      "target": "results",
      "type": "write",
      "label": "Write native acceptance report",
      "evidence": [
        {
          "path": "scripts/verify_native_hue_release.py",
          "symbol": "verify",
          "line": 122,
          "end_line": 269,
          "sha256": "6117f5fdcdcb1019e7c92b9f4b8d4183498ff355da43e2a2477418c903a50b4e",
          "url": "source.html?path=scripts/verify_native_hue_release.py#L122"
        }
      ],
      "boundary": true,
      "flows": [
        "release"
      ]
    },
    {
      "id": "provenance-legacy",
      "source": "provenance",
      "target": "legacy-gates",
      "type": "call",
      "label": "Legacy source and research evidence",
      "evidence": [
        {
          "path": "scripts/verify.py",
          "symbol": "verify",
          "line": 18,
          "end_line": 100,
          "sha256": "edb0c32960052ffa22d32d029aaf2fcac17ecf290e31efe34aa96c1a8d0cd30e",
          "url": "source.html?path=scripts/verify.py#L18"
        }
      ],
      "boundary": false,
      "flows": []
    },
    {
      "id": "legacy-results",
      "source": "legacy-gates",
      "target": "results",
      "type": "write",
      "label": "Write separate legacy acceptance report",
      "evidence": [
        {
          "path": "scripts/verify.py",
          "symbol": "main",
          "line": 103,
          "end_line": 121,
          "sha256": "edb0c32960052ffa22d32d029aaf2fcac17ecf290e31efe34aa96c1a8d0cd30e",
          "url": "source.html?path=scripts/verify.py#L103"
        }
      ],
      "boundary": true,
      "flows": []
    }
  ],
  "flows": [
    {
      "id": "native",
      "name": "Python hue operation",
      "description": "Import a native alias, create RGBD storage through the ABI, borrow operands and return a new Rust tensor.",
      "steps": [
        "package",
        "python",
        "ffi",
        "tokens",
        "tensor",
        "operators"
      ]
    },
    {
      "id": "save-load",
      "name": "Save and load native bytes",
      "description": "The binding passes a path and budget to Rust storage. Loading checks the file before allocating token storage.",
      "steps": [
        "python",
        "ffi",
        "storage",
        "tensor"
      ]
    },
    {
      "id": "cli",
      "name": "Standalone CLI",
      "description": "Command text reaches the Rust constructors, operators and file codec directly.",
      "steps": [
        "cli",
        "tensor",
        "operators",
        "storage"
      ]
    },
    {
      "id": "inspection",
      "name": "Inspect a legacy cube",
      "description": "The browser submits local JSON. The server validates authority and bounds, runs legacy codecs or cube operations and returns measured results.",
      "steps": [
        "browser",
        "http",
        "inspection",
        "codecs",
        "cubes",
        "results"
      ]
    },
    {
      "id": "training",
      "name": "Run legacy training",
      "description": "Seeded dataset selection feeds floating training and quantization-aware layers. Checkpoints and measured results retain their provenance.",
      "steps": [
        "datasets",
        "training",
        "nn",
        "legacy-storage",
        "results"
      ]
    },
    {
      "id": "release",
      "name": "Check a native release",
      "description": "Exact binding and header enter the wheel. The checker compares binaries, sources, integer oracle and isolated installation records.",
      "steps": [
        "python",
        "header",
        "build",
        "provenance",
        "acceptance",
        "results"
      ]
    }
  ],
  "integrations": [
    {
      "name": "Native Rust dependencies",
      "status": "present",
      "detail": "Standard library only. Cargo.toml declares no third-party dependencies.",
      "sources": [
        {
          "path": "native/hue_tensor/Cargo.toml",
          "line": 1,
          "end_line": 19,
          "sha256": "96bb117ad65e5beb74365b0f84289ba1f2a64a018ad05c7faf2de990a3f3bb9c",
          "url": "source.html?path=native/hue_tensor/Cargo.toml#L1"
        }
      ]
    },
    {
      "name": "Native Python dependencies",
      "status": "present",
      "detail": "The standalone chromatic-hue wheel uses ctypes and other standard-library modules. The full legacy package has separate dependencies.",
      "sources": [
        {
          "path": "chromatensor/hue.py",
          "line": 1,
          "end_line": 428,
          "sha256": "9781800dbbce9893283390ee3b409027f95d48a5fcef01e48864ea5cb3126c97",
          "url": "source.html?path=chromatensor/hue.py#L1"
        },
        {
          "path": "scripts/build_native_hue.py",
          "symbol": "make_wheel",
          "line": 86,
          "end_line": 126,
          "sha256": "354fd81cdef0aa748b7b3eea4adc8b630c3f0c939468c7f67b27ffe510ca9102",
          "url": "source.html?path=scripts/build_native_hue.py#L86"
        }
      ]
    },
    {
      "name": "PyTorch and NumPy",
      "status": "present",
      "detail": "Legacy numerical APIs and research use these libraries. They are not native ring execution.",
      "sources": [
        {
          "path": "pyproject.toml",
          "contains": "dependencies =",
          "line": 13,
          "end_line": 13,
          "sha256": "91eb7b747f849031fc2cd4515a73040364f1a9ac20d38c420cbcccadee2abc6c",
          "url": "source.html?path=pyproject.toml#L13"
        }
      ]
    },
    {
      "name": "Dataset downloads",
      "status": "present",
      "detail": "torchvision image downloads and Tiny Shakespeare URL retrieval may use the network. The architecture explorer itself performs no network requests.",
      "sources": [
        {
          "path": "experiments/datasets.py",
          "symbol": "image_data",
          "line": 42,
          "end_line": 63,
          "sha256": "d79df91b2ebad6bb51c02d0f96d69d2f0e674bfbecb8027441306910eb37e9f2",
          "url": "source.html?path=experiments/datasets.py#L42"
        },
        {
          "path": "experiments/datasets.py",
          "symbol": "language_data",
          "line": 74,
          "end_line": 96,
          "sha256": "d79df91b2ebad6bb51c02d0f96d69d2f0e674bfbecb8027441306910eb37e9f2",
          "url": "source.html?path=experiments/datasets.py#L74"
        }
      ]
    },
    {
      "name": "Cargo and container toolchain",
      "status": "present",
      "detail": "Cargo runs with --offline. The Linux cross-build uses a pinned Rust container, which must already be available or pulled by the container runtime.",
      "sources": [
        {
          "path": "scripts/build_native_hue.py",
          "symbol": "main",
          "line": 129,
          "end_line": 197,
          "sha256": "354fd81cdef0aa748b7b3eea4adc8b630c3f0c939468c7f67b27ffe510ca9102",
          "url": "source.html?path=scripts/build_native_hue.py#L129"
        }
      ]
    },
    {
      "name": "Accounts and authentication",
      "status": "absent",
      "detail": "No account model, sign-in flow or authorization service was found in the reviewed runtime. Local Host and Origin checks are not account authentication.",
      "sources": [
        {
          "path": "pyproject.toml",
          "line": 1,
          "end_line": 27,
          "sha256": "91eb7b747f849031fc2cd4515a73040364f1a9ac20d38c420cbcccadee2abc6c",
          "url": "source.html?path=pyproject.toml#L1"
        },
        {
          "path": "dashboard/server.py",
          "symbol": "DashboardHandler",
          "line": 129,
          "end_line": 280,
          "sha256": "c9ee7664b55deb67b03c6ed051ff3f8eeeca4576befce7cdd103ee871ccf2f2d",
          "url": "source.html?path=dashboard/server.py#L129"
        }
      ]
    },
    {
      "name": "Chat and messaging",
      "status": "absent",
      "detail": "No chat backend, message store or messaging integration was found.",
      "sources": [
        {
          "path": "pyproject.toml",
          "line": 1,
          "end_line": 27,
          "sha256": "91eb7b747f849031fc2cd4515a73040364f1a9ac20d38c420cbcccadee2abc6c",
          "url": "source.html?path=pyproject.toml#L1"
        },
        {
          "path": "dashboard/server.py",
          "symbol": "DashboardHandler",
          "line": 129,
          "end_line": 280,
          "sha256": "c9ee7664b55deb67b03c6ed051ff3f8eeeca4576befce7cdd103ee871ccf2f2d",
          "url": "source.html?path=dashboard/server.py#L129"
        }
      ]
    },
    {
      "name": "Payments and billing",
      "status": "absent",
      "detail": "No payment processor, checkout or billing code was found. Licensing terms do not implement payment collection.",
      "sources": [
        {
          "path": "pyproject.toml",
          "line": 1,
          "end_line": 27,
          "sha256": "91eb7b747f849031fc2cd4515a73040364f1a9ac20d38c420cbcccadee2abc6c",
          "url": "source.html?path=pyproject.toml#L1"
        },
        {
          "path": "dashboard/server.py",
          "symbol": "DashboardHandler",
          "line": 129,
          "end_line": 280,
          "sha256": "c9ee7664b55deb67b03c6ed051ff3f8eeeca4576befce7cdd103ee871ccf2f2d",
          "url": "source.html?path=dashboard/server.py#L129"
        }
      ]
    },
    {
      "name": "Convex or cloud database",
      "status": "absent",
      "detail": "No Convex client, schema or hosted database integration was found. Persistence uses local files.",
      "sources": [
        {
          "path": "pyproject.toml",
          "line": 1,
          "end_line": 27,
          "sha256": "91eb7b747f849031fc2cd4515a73040364f1a9ac20d38c420cbcccadee2abc6c",
          "url": "source.html?path=pyproject.toml#L1"
        },
        {
          "path": "dashboard/server.py",
          "symbol": "DashboardHandler",
          "line": 129,
          "end_line": 280,
          "sha256": "c9ee7664b55deb67b03c6ed051ff3f8eeeca4576befce7cdd103ee871ccf2f2d",
          "url": "source.html?path=dashboard/server.py#L129"
        }
      ]
    },
    {
      "name": "Hosted inference providers",
      "status": "absent",
      "detail": "No OpenAI, Anthropic or other hosted inference provider call was found. Legacy model inference runs locally through PyTorch.",
      "sources": [
        {
          "path": "pyproject.toml",
          "line": 1,
          "end_line": 27,
          "sha256": "91eb7b747f849031fc2cd4515a73040364f1a9ac20d38c420cbcccadee2abc6c",
          "url": "source.html?path=pyproject.toml#L1"
        },
        {
          "path": "dashboard/server.py",
          "symbol": "DashboardHandler",
          "line": 129,
          "end_line": 280,
          "sha256": "c9ee7664b55deb67b03c6ed051ff3f8eeeca4576befce7cdd103ee871ccf2f2d",
          "url": "source.html?path=dashboard/server.py#L129"
        }
      ]
    }
  ],
  "inventory": [
    {
      "path": ".github/workflows/ci.yml",
      "sha256": "00a3036dce03362859492782399a371d2422e2069425b01501c07c3b4c65e3c8",
      "imports": []
    },
    {
      "path": "chromatensor/__init__.py",
      "sha256": "5a0d9049c45cabb47f36f66a6800cd714d3b0ff8463f3a275f1eb9948be432c7",
      "imports": [
        {
          "module": "importlib",
          "line": 2
        },
        {
          "module": ".hue",
          "line": 4
        }
      ]
    },
    {
      "path": "chromatensor/autograd/__init__.py",
      "sha256": "29dbb3efe94820aa9b6d171c32fe220eb779f633c7c700309c7420a5e9f6f6ea",
      "imports": [
        {
          "module": ".quantization",
          "line": 3
        }
      ]
    },
    {
      "path": "chromatensor/autograd/quantization.py",
      "sha256": "9524cd0af99950a2852a84cadbd29b8d374a089dce01f793cf903da9e72fe393",
      "imports": [
        {
          "module": "__future__",
          "line": 8
        },
        {
          "module": "dataclasses",
          "line": 10
        },
        {
          "module": "math",
          "line": 11
        },
        {
          "module": "torch",
          "line": 13
        },
        {
          "module": "torch",
          "line": 14
        },
        {
          "module": "..codecs.rgbd",
          "line": 16
        }
      ]
    },
    {
      "path": "chromatensor/codecs/__init__.py",
      "sha256": "fc436cb22a165f6dfa2cc25efc6c6348d304eea468545edf3ee31e355797080e",
      "imports": [
        {
          "module": ".base",
          "line": 2
        },
        {
          "module": ".rgb",
          "line": 3
        },
        {
          "module": ".rgbd",
          "line": 4
        },
        {
          "module": ".hsv",
          "line": 5
        },
        {
          "module": ".learned",
          "line": 6
        },
        {
          "module": ".hierarchical",
          "line": 8
        },
        {
          "module": ".neural_rgb",
          "line": 9
        }
      ]
    },
    {
      "path": "chromatensor/codecs/base.py",
      "sha256": "51cf4214dcb731871ff218ff72d312569068ce13b4a0d70f0e2952651d68657b",
      "imports": [
        {
          "module": "__future__",
          "line": 2
        },
        {
          "module": "math",
          "line": 4
        },
        {
          "module": "typing",
          "line": 5
        },
        {
          "module": "torch",
          "line": 7
        }
      ]
    },
    {
      "path": "chromatensor/codecs/hierarchical.py",
      "sha256": "cef4692aecbc208564489bc165cc9b0c9ef1440257f735734326e792ecc1cec3",
      "imports": [
        {
          "module": "__future__",
          "line": 2
        },
        {
          "module": "math",
          "line": 4
        },
        {
          "module": "torch",
          "line": 6
        },
        {
          "module": ".base",
          "line": 8
        }
      ]
    },
    {
      "path": "chromatensor/codecs/hsv.py",
      "sha256": "5ce2eb66a2e80970f46602808b000a30d0cb4203611659029a5079792d97423d",
      "imports": [
        {
          "module": "__future__",
          "line": 2
        },
        {
          "module": "math",
          "line": 4
        },
        {
          "module": "torch",
          "line": 6
        },
        {
          "module": ".base",
          "line": 8
        }
      ]
    },
    {
      "path": "chromatensor/codecs/learned.py",
      "sha256": "1bcb2fdbfccc1f0423a7c4624476793875903c2d008fa696de1aa613e37a7281",
      "imports": [
        {
          "module": "__future__",
          "line": 2
        },
        {
          "module": "math",
          "line": 4
        },
        {
          "module": "torch",
          "line": 6
        },
        {
          "module": "torch",
          "line": 7
        },
        {
          "module": ".base",
          "line": 9
        }
      ]
    },
    {
      "path": "chromatensor/codecs/neural_rgb.py",
      "sha256": "156c782f66096f8960c863b8d743146f00cf05d219b419fcef7291762947b0da",
      "imports": [
        {
          "module": "__future__",
          "line": 2
        },
        {
          "module": "math",
          "line": 4
        },
        {
          "module": "torch",
          "line": 6
        },
        {
          "module": "torch",
          "line": 7
        },
        {
          "module": ".base",
          "line": 9
        }
      ]
    },
    {
      "path": "chromatensor/codecs/rgb.py",
      "sha256": "aa8ec8ed36333fe5f212441e9a288c8f1cc6254425c6de5278625ef89d3b9906",
      "imports": [
        {
          "module": "__future__",
          "line": 2
        },
        {
          "module": "torch",
          "line": 4
        },
        {
          "module": ".base",
          "line": 6
        }
      ]
    },
    {
      "path": "chromatensor/codecs/rgbd.py",
      "sha256": "0de8229189a88d4ef2806cf295a8f3bc1f11efe308f4aa1b166e944f9b273617",
      "imports": [
        {
          "module": "__future__",
          "line": 2
        },
        {
          "module": "torch",
          "line": 4
        },
        {
          "module": ".base",
          "line": 6
        }
      ]
    },
    {
      "path": "chromatensor/cube.py",
      "sha256": "743086b6121e60630681e4958a65f9bd2195e71ef04520ef43ad8fc2dbd24451",
      "imports": [
        {
          "module": "__future__",
          "line": 2
        },
        {
          "module": "collections.abc",
          "line": 4
        },
        {
          "module": "hashlib",
          "line": 5
        },
        {
          "module": "io",
          "line": 6
        },
        {
          "module": "json",
          "line": 7
        },
        {
          "module": "math",
          "line": 8
        },
        {
          "module": "operator",
          "line": 9
        },
        {
          "module": "pathlib",
          "line": 10
        },
        {
          "module": "zipfile",
          "line": 11
        },
        {
          "module": "numpy",
          "line": 13
        },
        {
          "module": "torch",
          "line": 14
        },
        {
          "module": ".codecs",
          "line": 16
        },
        {
          "module": ".codecs.base",
          "line": 17
        },
        {
          "module": ".serialization.format",
          "line": 18
        },
        {
          "module": ".tensor",
          "line": 22
        }
      ]
    },
    {
      "path": "chromatensor/decoders.py",
      "sha256": "70afdc08e3dd966dd023d1612574047b032399e0c1ec34feaeb5ecbfd6a0e2df",
      "imports": [
        {
          "module": "__future__",
          "line": 2
        },
        {
          "module": "math",
          "line": 4
        },
        {
          "module": "torch",
          "line": 6
        },
        {
          "module": "torch",
          "line": 7
        },
        {
          "module": ".codecs.base",
          "line": 9
        }
      ]
    },
    {
      "path": "chromatensor/holographic.py",
      "sha256": "379630fa90845a4a1ad12282edbd7fbad90403ee6f12cc70f25231530e4ff997",
      "imports": [
        {
          "module": "__future__",
          "line": 7
        },
        {
          "module": "collections.abc",
          "line": 9
        },
        {
          "module": "hashlib",
          "line": 10
        },
        {
          "module": "json",
          "line": 11
        },
        {
          "module": "math",
          "line": 12
        },
        {
          "module": "os",
          "line": 13
        },
        {
          "module": "pathlib",
          "line": 14
        },
        {
          "module": "struct",
          "line": 15
        },
        {
          "module": "tempfile",
          "line": 16
        },
        {
          "module": "zlib",
          "line": 17
        },
        {
          "module": "numpy",
          "line": 19
        },
        {
          "module": "torch",
          "line": 20
        },
        {
          "module": ".codecs.hsv",
          "line": 22
        },
        {
          "module": ".serialization.format",
          "line": 23
        }
      ]
    },
    {
      "path": "chromatensor/hue.py",
      "sha256": "9781800dbbce9893283390ee3b409027f95d48a5fcef01e48864ea5cb3126c97",
      "imports": [
        {
          "module": "__future__",
          "line": 2
        },
        {
          "module": "ctypes",
          "line": 4
        },
        {
          "module": "contextlib",
          "line": 5
        },
        {
          "module": "hashlib",
          "line": 6
        },
        {
          "module": "importlib.util",
          "line": 7
        },
        {
          "module": "math",
          "line": 8
        },
        {
          "module": "operator",
          "line": 9
        },
        {
          "module": "os",
          "line": 10
        },
        {
          "module": "pathlib",
          "line": 11
        },
        {
          "module": "platform",
          "line": 12
        },
        {
          "module": "threading",
          "line": 13
        },
        {
          "module": "functools",
          "line": 14
        }
      ]
    },
    {
      "path": "chromatensor/nn/__init__.py",
      "sha256": "d0037299d4799f5434fe1c315e93fc43bed88c14b69f5d5ee52943c04bc162e6",
      "imports": [
        {
          "module": ".attention",
          "line": 3
        },
        {
          "module": ".layers",
          "line": 4
        },
        {
          "module": ".report",
          "line": 5
        }
      ]
    },
    {
      "path": "chromatensor/nn/attention.py",
      "sha256": "1384f870058192eb4b321bce66698dfce5b2383cf911a5417439af3954f8dd1f",
      "imports": [
        {
          "module": "__future__",
          "line": 3
        },
        {
          "module": "math",
          "line": 5
        },
        {
          "module": "torch",
          "line": 7
        },
        {
          "module": "torch",
          "line": 8
        },
        {
          "module": "torch.nn",
          "line": 9
        },
        {
          "module": ".layers",
          "line": 11
        }
      ]
    },
    {
      "path": "chromatensor/nn/layers.py",
      "sha256": "2570772713de51c263c5ffd653d61e0f535dcb3b1b9ffc8501ae6e7e9ccde194",
      "imports": [
        {
          "module": "__future__",
          "line": 7
        },
        {
          "module": "copy",
          "line": 9
        },
        {
          "module": "math",
          "line": 10
        },
        {
          "module": "collections.abc",
          "line": 11
        },
        {
          "module": "torch",
          "line": 13
        },
        {
          "module": "torch",
          "line": 14
        },
        {
          "module": "torch.nn",
          "line": 15
        },
        {
          "module": "..autograd",
          "line": 17
        }
      ]
    },
    {
      "path": "chromatensor/nn/report.py",
      "sha256": "ca387c1d8048a3d08637740bfc2ef778f9fd243b08bede4d905cbcaff6e486ac",
      "imports": [
        {
          "module": "__future__",
          "line": 3
        },
        {
          "module": "torch",
          "line": 5
        },
        {
          "module": "..autograd",
          "line": 7
        },
        {
          "module": ".layers",
          "line": 8
        }
      ]
    },
    {
      "path": "chromatensor/operators/__init__.py",
      "sha256": "a72042a18d3b62e5224a7cd034b26ac317d5299d791717bde9d8e0b49abed740",
      "imports": [
        {
          "module": ".core",
          "line": 2
        }
      ]
    },
    {
      "path": "chromatensor/operators/core.py",
      "sha256": "5db3a66e45c15f764845c71d92053f1d60f0d6bd7c6632b0d3fcabf6d47c35a7",
      "imports": [
        {
          "module": "__future__",
          "line": 1
        },
        {
          "module": "torch",
          "line": 3
        },
        {
          "module": "torch",
          "line": 4
        },
        {
          "module": "..tensor",
          "line": 5
        },
        {
          "module": "..codecs",
          "line": 6
        },
        {
          "module": "..codecs.base",
          "line": 7
        }
      ]
    },
    {
      "path": "chromatensor/serialization/__init__.py",
      "sha256": "0052c5e4c372360c4fab24b9885a6fb1f5fb344d972e886acbe40c19c6764e2c",
      "imports": [
        {
          "module": ".format",
          "line": 2
        }
      ]
    },
    {
      "path": "chromatensor/serialization/format.py",
      "sha256": "6b2694152b7a2d1face0eb6db891ac6effea9a0d06287657b63be97e9936e8b1",
      "imports": [
        {
          "module": "__future__",
          "line": 1
        },
        {
          "module": "hashlib",
          "line": 3
        },
        {
          "module": "contextlib",
          "line": 4
        },
        {
          "module": "io",
          "line": 5
        },
        {
          "module": "json",
          "line": 6
        },
        {
          "module": "math",
          "line": 7
        },
        {
          "module": "os",
          "line": 8
        },
        {
          "module": "pathlib",
          "line": 9
        },
        {
          "module": "struct",
          "line": 10
        },
        {
          "module": "tempfile",
          "line": 11
        },
        {
          "module": "zipfile",
          "line": 12
        },
        {
          "module": "zlib",
          "line": 13
        },
        {
          "module": "numpy",
          "line": 15
        },
        {
          "module": "torch",
          "line": 16
        },
        {
          "module": "..codecs",
          "line": 173
        },
        {
          "module": "..tensor",
          "line": 174
        },
        {
          "module": "..codecs",
          "line": 241
        },
        {
          "module": "..tensor",
          "line": 242
        }
      ]
    },
    {
      "path": "chromatensor/tensor.py",
      "sha256": "7daff1f6685c23d2713a2879f18480d1097d878af50c82487628486ac8f72ed8",
      "imports": [
        {
          "module": "__future__",
          "line": 2
        },
        {
          "module": "math",
          "line": 4
        },
        {
          "module": "collections.abc",
          "line": 5
        },
        {
          "module": "numpy",
          "line": 7
        },
        {
          "module": "torch",
          "line": 8
        },
        {
          "module": ".codecs",
          "line": 10
        },
        {
          "module": ".codecs.base",
          "line": 11
        }
      ]
    },
    {
      "path": "chromatensor/tokenizer/__init__.py",
      "sha256": "a3b4a1b7203fc8d4c52fe9a1b03c3d9eee3dcc78d7d82891f447f9fc98273e72",
      "imports": [
        {
          "module": "__future__",
          "line": 2
        },
        {
          "module": "math",
          "line": 4
        },
        {
          "module": "dataclasses",
          "line": 5
        },
        {
          "module": "torch",
          "line": 7
        },
        {
          "module": "..codecs",
          "line": 9
        },
        {
          "module": "..codecs.base",
          "line": 11
        },
        {
          "module": "..tensor",
          "line": 169
        },
        {
          "module": "..serialization",
          "line": 170
        },
        {
          "module": "..serialization",
          "line": 178
        }
      ]
    },
    {
      "path": "chromatensor/visualization/__init__.py",
      "sha256": "fd1b9dbc0ae4007762a7bf237ede380fc6e2a8693dc65704b45dade8322d18a7",
      "imports": [
        {
          "module": ".plots",
          "line": 2
        }
      ]
    },
    {
      "path": "chromatensor/visualization/plots.py",
      "sha256": "6ec35b91cf7769bff87e86743f8e985f40b3d097153a57420392389abcd6a3ff",
      "imports": [
        {
          "module": "__future__",
          "line": 1
        },
        {
          "module": "math",
          "line": 3
        },
        {
          "module": "pathlib",
          "line": 4
        },
        {
          "module": "numpy",
          "line": 5
        },
        {
          "module": "matplotlib",
          "line": 6
        },
        {
          "module": "matplotlib.pyplot",
          "line": 8
        }
      ]
    },
    {
      "path": "dashboard/__init__.py",
      "sha256": "e795b4116ee4f719fdc3731d6de0bc49c19ae03bac6558916ac8a79407f724a9",
      "imports": []
    },
    {
      "path": "dashboard/app.js",
      "sha256": "59096088da2640564034b45108d19cd70477c3affa02954b30a03773fd7bb9a7",
      "imports": []
    },
    {
      "path": "dashboard/cube.css",
      "sha256": "5653d9c8a21e26321d6100b67f42429bbe97b4cab35b82151e76c3128b4d5ff5",
      "imports": []
    },
    {
      "path": "dashboard/cube.js",
      "sha256": "22dae4dcecbf2e72e4c9140596950b8436910797a82c4646b279f21ea3804355",
      "imports": []
    },
    {
      "path": "dashboard/cube_api.py",
      "sha256": "30a6099e2f2726c498f300d213e0ad55c8ed8f2bf68b49597cfe247fbe58f472",
      "imports": [
        {
          "module": "__future__",
          "line": 2
        },
        {
          "module": "hashlib",
          "line": 4
        },
        {
          "module": "itertools",
          "line": 5
        },
        {
          "module": "math",
          "line": 6
        },
        {
          "module": "pathlib",
          "line": 7
        },
        {
          "module": "json",
          "line": 8
        },
        {
          "module": "torch",
          "line": 10
        },
        {
          "module": "chromatensor",
          "line": 26
        },
        {
          "module": "chromatensor.codecs",
          "line": 27
        },
        {
          "module": "chromatensor.cube",
          "line": 28
        },
        {
          "module": "chromatensor",
          "line": 71
        },
        {
          "module": "chromatensor.cube",
          "line": 72
        },
        {
          "module": "chromatensor.holographic",
          "line": 195
        },
        {
          "module": "chromatensor.codecs.hsv",
          "line": 196
        }
      ]
    },
    {
      "path": "dashboard/index.html",
      "sha256": "6f8ac1d22a78af2d9b64e795a96b5e4ad1e4d7edf67cd7e41871ba997abd0c0b",
      "imports": []
    },
    {
      "path": "dashboard/server.py",
      "sha256": "c9ee7664b55deb67b03c6ed051ff3f8eeeca4576befce7cdd103ee871ccf2f2d",
      "imports": [
        {
          "module": "__future__",
          "line": 3
        },
        {
          "module": "argparse",
          "line": 5
        },
        {
          "module": "colorsys",
          "line": 6
        },
        {
          "module": "json",
          "line": 7
        },
        {
          "module": "math",
          "line": 8
        },
        {
          "module": "mimetypes",
          "line": 9
        },
        {
          "module": "ipaddress",
          "line": 10
        },
        {
          "module": "socket",
          "line": 11
        },
        {
          "module": "time",
          "line": 12
        },
        {
          "module": "collections",
          "line": 13
        },
        {
          "module": "http.server",
          "line": 14
        },
        {
          "module": "pathlib",
          "line": 15
        },
        {
          "module": "urllib.parse",
          "line": 16
        },
        {
          "module": "torch",
          "line": 27
        },
        {
          "module": "chromatensor.tensor",
          "line": 28
        },
        {
          "module": "chromatensor.codecs",
          "line": 29
        },
        {
          "module": ".cube_api",
          "line": 270
        }
      ]
    },
    {
      "path": "dashboard/style.css",
      "sha256": "80850bbe4417bcf3e6eea8b0417cf58f7da920671e7ff7e45fc29036f15e7bf3",
      "imports": []
    },
    {
      "path": "experiments/__init__.py",
      "sha256": "75a49e4b275144336821845efcd63262c51331b2572e05551fa8cb23bf0de8c6",
      "imports": []
    },
    {
      "path": "experiments/ablations.py",
      "sha256": "8af0ad62ac5a1488e582d2fabc68698bf7316ca04c131a411d8ed065fdb2d65d",
      "imports": [
        {
          "module": "__future__",
          "line": 2
        },
        {
          "module": "torch",
          "line": 4
        },
        {
          "module": "chromatensor",
          "line": 5
        },
        {
          "module": "chromatensor.autograd",
          "line": 6
        },
        {
          "module": ".common",
          "line": 7
        }
      ]
    },
    {
      "path": "experiments/attention.py",
      "sha256": "c7f73ab6df16cda03f9d64044c3f99c86d793b7d7ec02f8337728b2c4bc638c7",
      "imports": [
        {
          "module": "__future__",
          "line": 3
        },
        {
          "module": "os",
          "line": 5
        },
        {
          "module": "argparse",
          "line": 8
        },
        {
          "module": "math",
          "line": 9
        },
        {
          "module": "pathlib",
          "line": 10
        },
        {
          "module": "time",
          "line": 11
        },
        {
          "module": "torch",
          "line": 13
        },
        {
          "module": "torch",
          "line": 14
        },
        {
          "module": "chromatensor",
          "line": 16
        },
        {
          "module": "chromatensor.nn",
          "line": 17
        },
        {
          "module": ".common",
          "line": 18
        },
        {
          "module": ".training",
          "line": 19
        },
        {
          "module": ".provenance",
          "line": 240
        },
        {
          "module": ".checkpoint_probe",
          "line": 218
        }
      ]
    },
    {
      "path": "experiments/checkpoint_probe.py",
      "sha256": "07b8db5e577b0d06ac746c42bf8bb903d201a54c37a11f9c8634541826a8ca48",
      "imports": [
        {
          "module": "__future__",
          "line": 3
        },
        {
          "module": "contextlib",
          "line": 5
        },
        {
          "module": "hashlib",
          "line": 6
        },
        {
          "module": "io",
          "line": 7
        },
        {
          "module": "json",
          "line": 8
        },
        {
          "module": "math",
          "line": 9
        },
        {
          "module": "pathlib",
          "line": 10
        },
        {
          "module": "zipfile",
          "line": 11
        },
        {
          "module": "numpy",
          "line": 13
        },
        {
          "module": "torch",
          "line": 14
        },
        {
          "module": "chromatensor",
          "line": 16
        },
        {
          "module": "chromatensor.serialization.format",
          "line": 17
        },
        {
          "module": ".common",
          "line": 18
        },
        {
          "module": ".attention",
          "line": 107
        },
        {
          "module": "chromatensor.codecs",
          "line": 111
        },
        {
          "module": ".training",
          "line": 115
        }
      ]
    },
    {
      "path": "experiments/common.py",
      "sha256": "6842cdc94effcf926ef8f8d46f409ee91b05a507a8c671555de6f1b18a8b63ad",
      "imports": [
        {
          "module": "__future__",
          "line": 1
        },
        {
          "module": "contextlib",
          "line": 3
        },
        {
          "module": "datetime",
          "line": 4
        },
        {
          "module": "hashlib",
          "line": 5
        },
        {
          "module": "json",
          "line": 6
        },
        {
          "module": "os",
          "line": 7
        },
        {
          "module": "pathlib",
          "line": 8
        },
        {
          "module": "platform",
          "line": 9
        },
        {
          "module": "random",
          "line": 10
        },
        {
          "module": "tempfile",
          "line": 11
        },
        {
          "module": "threading",
          "line": 12
        },
        {
          "module": "time",
          "line": 13
        },
        {
          "module": "numpy",
          "line": 15
        },
        {
          "module": "psutil",
          "line": 16
        },
        {
          "module": "torch",
          "line": 17
        },
        {
          "module": ".provenance",
          "line": 31
        },
        {
          "module": ".provenance",
          "line": 102
        }
      ]
    },
    {
      "path": "experiments/cube.py",
      "sha256": "745371c258c9fd10e50d753aea457e9d5305950d2e7fa7befb9583e7725860fe",
      "imports": [
        {
          "module": "__future__",
          "line": 2
        },
        {
          "module": "math",
          "line": 4
        },
        {
          "module": "pathlib",
          "line": 5
        },
        {
          "module": "time",
          "line": 6
        },
        {
          "module": "torch",
          "line": 8
        },
        {
          "module": "chromatensor",
          "line": 10
        },
        {
          "module": "chromatensor.nn",
          "line": 11
        },
        {
          "module": ".common",
          "line": 12
        },
        {
          "module": ".models",
          "line": 13
        },
        {
          "module": ".provenance",
          "line": 83
        },
        {
          "module": "chromatensor.holographic",
          "line": 179
        }
      ]
    },
    {
      "path": "experiments/datasets.py",
      "sha256": "d79df91b2ebad6bb51c02d0f96d69d2f0e674bfbecb8027441306910eb37e9f2",
      "imports": [
        {
          "module": "__future__",
          "line": 1
        },
        {
          "module": "hashlib",
          "line": 3
        },
        {
          "module": "pathlib",
          "line": 4
        },
        {
          "module": "urllib.request",
          "line": 5
        },
        {
          "module": "numpy",
          "line": 6
        },
        {
          "module": "torch",
          "line": 7
        },
        {
          "module": ".common",
          "line": 8
        },
        {
          "module": "torchvision.datasets",
          "line": 43
        }
      ]
    },
    {
      "path": "experiments/decoder.py",
      "sha256": "152913a4d0e58412b95c7a6f0c1c6dca5a76fefc3c1f033beb6ff18cdd3c2515",
      "imports": [
        {
          "module": "__future__",
          "line": 2
        },
        {
          "module": "json",
          "line": 4
        },
        {
          "module": "pathlib",
          "line": 5
        },
        {
          "module": "time",
          "line": 6
        },
        {
          "module": "numpy",
          "line": 8
        },
        {
          "module": "torch",
          "line": 9
        },
        {
          "module": "chromatensor.codecs",
          "line": 11
        },
        {
          "module": "chromatensor.decoders",
          "line": 12
        },
        {
          "module": ".common",
          "line": 13
        },
        {
          "module": ".training",
          "line": 14
        },
        {
          "module": ".checkpoint_probe",
          "line": 149
        }
      ]
    },
    {
      "path": "experiments/instrumentation.py",
      "sha256": "7b0f859324fc8d57ba82850aa0264132a8dd3b3acf8b6af0ca752a2c918d90e2",
      "imports": [
        {
          "module": "__future__",
          "line": 3
        },
        {
          "module": "io",
          "line": 5
        },
        {
          "module": "math",
          "line": 6
        },
        {
          "module": "pathlib",
          "line": 7
        },
        {
          "module": "zipfile",
          "line": 8
        },
        {
          "module": "numpy",
          "line": 10
        },
        {
          "module": "torch",
          "line": 11
        },
        {
          "module": "chromatensor",
          "line": 13
        },
        {
          "module": "chromatensor.autograd",
          "line": 14
        },
        {
          "module": "chromatensor.nn",
          "line": 15
        },
        {
          "module": ".attention",
          "line": 16
        },
        {
          "module": ".training",
          "line": 17
        }
      ]
    },
    {
      "path": "experiments/models.py",
      "sha256": "6b8ac1333d55c5fdf49ae30475a1fcb12bea417b34c7a38a8af1d63015d0384b",
      "imports": [
        {
          "module": "__future__",
          "line": 1
        },
        {
          "module": "math",
          "line": 3
        },
        {
          "module": "torch",
          "line": 4
        },
        {
          "module": "torch",
          "line": 5
        },
        {
          "module": "copy",
          "line": 63
        }
      ]
    },
    {
      "path": "experiments/precision.py",
      "sha256": "97d5b673789297f808cce59fe80541e6d766f5ccb46497bf687c2047d01c0b48",
      "imports": [
        {
          "module": "__future__",
          "line": 1
        },
        {
          "module": "io",
          "line": 3
        },
        {
          "module": "json",
          "line": 4
        },
        {
          "module": "math",
          "line": 5
        },
        {
          "module": "pathlib",
          "line": 6
        },
        {
          "module": "time",
          "line": 7
        },
        {
          "module": "numpy",
          "line": 9
        },
        {
          "module": "torch",
          "line": 10
        },
        {
          "module": "torch",
          "line": 11
        },
        {
          "module": "chromatensor",
          "line": 13
        },
        {
          "module": "chromatensor.codecs",
          "line": 14
        },
        {
          "module": "chromatensor.operators",
          "line": 15
        },
        {
          "module": ".common",
          "line": 16
        },
        {
          "module": ".training",
          "line": 17
        }
      ]
    },
    {
      "path": "experiments/provenance.py",
      "sha256": "abe3a22747f65c04133ed253425b2fe1b9811ddf9f2a0fe804b72fd143d6eb5f",
      "imports": [
        {
          "module": "__future__",
          "line": 2
        },
        {
          "module": "datetime",
          "line": 4
        },
        {
          "module": "hashlib",
          "line": 5
        },
        {
          "module": "importlib",
          "line": 6
        },
        {
          "module": "json",
          "line": 7
        },
        {
          "module": "pathlib",
          "line": 8
        },
        {
          "module": "platform",
          "line": 9
        },
        {
          "module": "re",
          "line": 10
        },
        {
          "module": "subprocess",
          "line": 11
        },
        {
          "module": "uuid",
          "line": 12
        }
      ]
    },
    {
      "path": "experiments/reporting.py",
      "sha256": "0b4a0ac3f82a149dc3258b8c000b8530115b3ce4982bb650163b42e6d71027ec",
      "imports": [
        {
          "module": "__future__",
          "line": 1
        },
        {
          "module": "collections",
          "line": 3
        },
        {
          "module": "json",
          "line": 4
        },
        {
          "module": "pathlib",
          "line": 5
        },
        {
          "module": "statistics",
          "line": 6
        }
      ]
    },
    {
      "path": "experiments/training.py",
      "sha256": "936c0791667a84e9b1aed08966c5ba090454ad87c5bf31a082ac2859553bb614",
      "imports": [
        {
          "module": "__future__",
          "line": 1
        },
        {
          "module": "gc",
          "line": 3
        },
        {
          "module": "io",
          "line": 4
        },
        {
          "module": "math",
          "line": 5
        },
        {
          "module": "pathlib",
          "line": 6
        },
        {
          "module": "time",
          "line": 7
        },
        {
          "module": "numpy",
          "line": 9
        },
        {
          "module": "torch",
          "line": 10
        },
        {
          "module": "torch",
          "line": 11
        },
        {
          "module": "chromatensor",
          "line": 13
        },
        {
          "module": "chromatensor.nn",
          "line": 14
        },
        {
          "module": ".common",
          "line": 15
        },
        {
          "module": ".datasets",
          "line": 16
        },
        {
          "module": ".models",
          "line": 17
        },
        {
          "module": ".checkpoint_probe",
          "line": 221
        },
        {
          "module": ".common",
          "line": 249
        }
      ]
    },
    {
      "path": "experiments/verification.py",
      "sha256": "7ccc8e9bd22c7d98bb13c97fa52ca6f9dcdab99853c7d80a4a6411f898dff317",
      "imports": [
        {
          "module": "__future__",
          "line": 2
        },
        {
          "module": "json",
          "line": 4
        },
        {
          "module": "hashlib",
          "line": 5
        },
        {
          "module": "math",
          "line": 6
        },
        {
          "module": "email",
          "line": 7
        },
        {
          "module": "email.parser",
          "line": 8
        },
        {
          "module": "pathlib",
          "line": 9
        },
        {
          "module": "xml.etree.ElementTree",
          "line": 10
        },
        {
          "module": "zipfile",
          "line": 11
        },
        {
          "module": "numpy",
          "line": 13
        },
        {
          "module": "torch",
          "line": 14
        },
        {
          "module": "chromatensor",
          "line": 16
        },
        {
          "module": ".common",
          "line": 17
        },
        {
          "module": ".provenance",
          "line": 18
        },
        {
          "module": ".reporting",
          "line": 151
        },
        {
          "module": ".checkpoint_probe",
          "line": 184
        },
        {
          "module": ".checkpoint_probe",
          "line": 190
        },
        {
          "module": ".models",
          "line": 197
        },
        {
          "module": ".instrumentation",
          "line": 224
        },
        {
          "module": "chromatensor.nn",
          "line": 251
        },
        {
          "module": ".cube",
          "line": 252
        },
        {
          "module": ".models",
          "line": 253
        },
        {
          "module": ".instrumentation",
          "line": 211
        }
      ]
    },
    {
      "path": "native/hue_tensor/Cargo.lock",
      "sha256": "1d70da3463b422e0470bdab478543cda8d58467f2fb2191fa780820c3a9e50a1",
      "imports": []
    },
    {
      "path": "native/hue_tensor/Cargo.toml",
      "sha256": "96bb117ad65e5beb74365b0f84289ba1f2a64a018ad05c7faf2de990a3f3bb9c",
      "imports": []
    },
    {
      "path": "native/hue_tensor/include/hue_tensor.h",
      "sha256": "81eaefa1bc501b8de9f96875a5a1723799db2532e8ba7b07bf7d6cd0c63a3f6c",
      "imports": []
    },
    {
      "path": "native/hue_tensor/src/bin/hue_tensor.rs",
      "sha256": "46438f9dc38d96ab1703875a883c8993fccdf77430851bda22746a4c03d3650c",
      "imports": [
        {
          "module": "use hue_tensor::storage::{load_tensor, save_tensor};",
          "line": 3
        },
        {
          "module": "use hue_tensor::{Error, Result, Tensor, Token, MODULUS};",
          "line": 4
        },
        {
          "module": "use std::path::Path;",
          "line": 5
        },
        {
          "module": "use std::sync::atomic::{AtomicU64, Ordering};",
          "line": 6
        },
        {
          "module": "use std::time::{SystemTime, UNIX_EPOCH};",
          "line": 7
        },
        {
          "module": "use super::*;",
          "line": 330
        }
      ]
    },
    {
      "path": "native/hue_tensor/src/ffi.rs",
      "sha256": "cd4e88e8cdd9be41c9196d3f8e22baf82c376398290ddf0c7f0b1ecf2144f920",
      "imports": [
        {
          "module": "use crate::{Error, Result, Tensor, Token, ABI_VERSION, MAX_ELEMENTS, MAX_RANK, MODULUS};",
          "line": 4
        },
        {
          "module": "use std::cell::RefCell;",
          "line": 5
        },
        {
          "module": "use std::ffi::{CStr, CString};",
          "line": 6
        },
        {
          "module": "use std::os::raw::c_char;",
          "line": 7
        },
        {
          "module": "use std::panic::{catch_unwind, AssertUnwindSafe};",
          "line": 8
        },
        {
          "module": "use std::path::Path;",
          "line": 9
        },
        {
          "module": "use std::ptr;",
          "line": 10
        },
        {
          "module": "use super::*;",
          "line": 525
        }
      ]
    },
    {
      "path": "native/hue_tensor/src/lib.rs",
      "sha256": "ab8c42dbcd5750c20f16534c6adf97a3a11b501e898b0583a1fd897abc6c0ccb",
      "imports": [
        {
          "module": "use std::fmt;",
          "line": 8
        },
        {
          "module": "use super::*;",
          "line": 471
        }
      ]
    },
    {
      "path": "native/hue_tensor/src/storage.rs",
      "sha256": "204dcf09f6f17e6f64b806aa157f9b1af207a0b3cd0c3ec78d6661adca11c137",
      "imports": [
        {
          "module": "use crate::{Error, Result, Tensor, Token, MAX_ELEMENTS, MAX_RANK};",
          "line": 7
        },
        {
          "module": "use std::ffi::OsString;",
          "line": 8
        },
        {
          "module": "use std::fs::{self, File, OpenOptions};",
          "line": 9
        },
        {
          "module": "use std::io::{Read, Write};",
          "line": 10
        },
        {
          "module": "use std::path::{Path, PathBuf};",
          "line": 11
        },
        {
          "module": "use std::sync::atomic::{AtomicU64, Ordering};",
          "line": 12
        },
        {
          "module": "use std::time::{SystemTime, UNIX_EPOCH};",
          "line": 13
        },
        {
          "module": "use super::*;",
          "line": 391
        }
      ]
    },
    {
      "path": "native/rust-notices/COPYRIGHT.html",
      "sha256": "139ebc62da648fa8dfea9d11ca6919ac417e8bfaffa08795341a414097a1fe24",
      "imports": []
    },
    {
      "path": "native/rust-notices/licenses/Apache-2.0.txt",
      "sha256": "074e6e32c86a4c0ef8b3ed25b721ca23aca83df277cd88106ef7177c354615ff",
      "imports": []
    },
    {
      "path": "native/rust-notices/licenses/BSD-2-Clause.txt",
      "sha256": "f32fb3b417a194167cfad068223fc975ba96c5960513a10f66a3c28720aec1df",
      "imports": []
    },
    {
      "path": "native/rust-notices/licenses/CC-BY-SA-4.0.txt",
      "sha256": "23ee78c8bae49cf08ea2f0c84945c66b987ebe4520881fb51b3dad4fb43d07c2",
      "imports": []
    },
    {
      "path": "native/rust-notices/licenses/GCC-exception-3.1.txt",
      "sha256": "a3966c9410ef94a1efbfa9ec04e9c1994026a70623fbf0885ce36737bfba3b54",
      "imports": []
    },
    {
      "path": "native/rust-notices/licenses/GPL-2.0-only.txt",
      "sha256": "54b212ba1f97f6948fa5486239ab505dd820b02e50cbfd296b4062c8609a70af",
      "imports": []
    },
    {
      "path": "native/rust-notices/licenses/GPL-3.0-or-later.txt",
      "sha256": "05315fce6c35f79b8a35454828995fa26336f235895ac64557abc028f4ffcea2",
      "imports": []
    },
    {
      "path": "native/rust-notices/licenses/ISC.txt",
      "sha256": "0ff61769b238634e042ab759dddfd104a8e69d091e3600ae3410e401a6164381",
      "imports": []
    },
    {
      "path": "native/rust-notices/licenses/LLVM-exception.txt",
      "sha256": "e34c58338bd89d43e709e226610d8f32b3e3c47f4ad9a99a8dc1d4ac7842488e",
      "imports": []
    },
    {
      "path": "native/rust-notices/licenses/MIT.txt",
      "sha256": "b85dcd3e453d05982552c52b5fc9e0bdd6d23c6f8e844b984a88af32570b0cc0",
      "imports": []
    },
    {
      "path": "native/rust-notices/licenses/NCSA.txt",
      "sha256": "39cb0234065c92b9e6cee3830cbb52b1aa289aa8fe9ba2e66730fec5cf05c630",
      "imports": []
    },
    {
      "path": "native/rust-notices/licenses/OFL-1.1.txt",
      "sha256": "8eea8287e5876b539670cadb82e99f9a7afddec6f6730811be1daf25d2e9bcfd",
      "imports": []
    },
    {
      "path": "native/rust-notices/licenses/Unicode-3.0.txt",
      "sha256": "f5062c9a188d81dfe66b56db4182dcf9e4b17c0d9b0d311a8e20b3a1b075c443",
      "imports": []
    },
    {
      "path": "native/rust-notices/manifest.json",
      "sha256": "8925e3c50c058f1ed823c350095878faf4db626835cf2615504dff4f90af5576",
      "imports": []
    },
    {
      "path": "native/rust-notices/README.md",
      "sha256": "8e5265971314eb6eb7bb6bc6bdba0a072c6455464fbb72093ec5df3f5094e5a7",
      "imports": []
    },
    {
      "path": "pyproject.toml",
      "sha256": "91eb7b747f849031fc2cd4515a73040364f1a9ac20d38c420cbcccadee2abc6c",
      "imports": []
    },
    {
      "path": "requirements/docs.txt",
      "sha256": "c30202fbc2f2eb1da451b8bae46b6867c07915844cec203a243991d14b8d4efc",
      "imports": []
    },
    {
      "path": "requirements/linux311-cpu.lock",
      "sha256": "354b5961c426ef7f16b3e6d13b3ff0de7d20541912542e5d2219535c83f90bc6",
      "imports": []
    },
    {
      "path": "requirements/linux312-cpu.lock",
      "sha256": "68de177a549e9ef146c76533bc897f969dccbb9f372d519c15757d2b280f5cee",
      "imports": []
    },
    {
      "path": "requirements/linux313-cpu.lock",
      "sha256": "68de177a549e9ef146c76533bc897f969dccbb9f372d519c15757d2b280f5cee",
      "imports": []
    },
    {
      "path": "requirements/linux314-cpu.lock",
      "sha256": "68de177a549e9ef146c76533bc897f969dccbb9f372d519c15757d2b280f5cee",
      "imports": []
    },
    {
      "path": "requirements/minimum-cpu.lock",
      "sha256": "297b5b4b73361d65d71163b6cbac1226836972890a3df79bfa81c4c67cbc6938",
      "imports": []
    },
    {
      "path": "requirements/minimum.txt",
      "sha256": "74b871ec1dd7eea38ac71eba810bd25e0c0d565c08f21538fb83328048abab54",
      "imports": []
    },
    {
      "path": "requirements/python311-cpu.lock",
      "sha256": "297b5b4b73361d65d71163b6cbac1226836972890a3df79bfa81c4c67cbc6938",
      "imports": []
    },
    {
      "path": "requirements/python312-cpu.lock",
      "sha256": "6d08237ffcfeedc2908dc42f9a99fa0e79f042fe01e4692984ff38c6f99b1a88",
      "imports": []
    },
    {
      "path": "requirements/python313-cpu.lock",
      "sha256": "6d08237ffcfeedc2908dc42f9a99fa0e79f042fe01e4692984ff38c6f99b1a88",
      "imports": []
    },
    {
      "path": "requirements/python314-cpu.lock",
      "sha256": "6d08237ffcfeedc2908dc42f9a99fa0e79f042fe01e4692984ff38c6f99b1a88",
      "imports": []
    },
    {
      "path": "scripts/architecture.py",
      "sha256": "6095e872c408855ceaeac6db2e891e2f5b5e9068d10422a067809ef1edd2d1f7",
      "imports": [
        {
          "module": "__future__",
          "line": 6
        },
        {
          "module": "argparse",
          "line": 8
        },
        {
          "module": "ast",
          "line": 9
        },
        {
          "module": "hashlib",
          "line": 10
        },
        {
          "module": "json",
          "line": 11
        },
        {
          "module": "pathlib",
          "line": 12
        },
        {
          "module": "re",
          "line": 13
        },
        {
          "module": "sys",
          "line": 14
        }
      ]
    },
    {
      "path": "scripts/build_docs_site.py",
      "sha256": "911e15d00dc3bf849a129b64cb12911281f4def70ed8b1a4267e2929374ffa60",
      "imports": [
        {
          "module": "__future__",
          "line": 2
        },
        {
          "module": "hashlib",
          "line": 4
        },
        {
          "module": "html",
          "line": 5
        },
        {
          "module": "html.parser",
          "line": 6
        },
        {
          "module": "json",
          "line": 7
        },
        {
          "module": "pathlib",
          "line": 8
        },
        {
          "module": "posixpath",
          "line": 9
        },
        {
          "module": "re",
          "line": 10
        },
        {
          "module": "shutil",
          "line": 11
        },
        {
          "module": "tempfile",
          "line": 12
        },
        {
          "module": "urllib.parse",
          "line": 13
        },
        {
          "module": "markdown",
          "line": 15
        },
        {
          "module": "release_audit",
          "line": 17
        },
        {
          "module": "scan_release",
          "line": 18
        }
      ]
    },
    {
      "path": "scripts/build_native_hue.py",
      "sha256": "354fd81cdef0aa748b7b3eea4adc8b630c3f0c939468c7f67b27ffe510ca9102",
      "imports": [
        {
          "module": "__future__",
          "line": 2
        },
        {
          "module": "argparse",
          "line": 4
        },
        {
          "module": "base64",
          "line": 5
        },
        {
          "module": "csv",
          "line": 6
        },
        {
          "module": "hashlib",
          "line": 7
        },
        {
          "module": "io",
          "line": 8
        },
        {
          "module": "json",
          "line": 9
        },
        {
          "module": "os",
          "line": 10
        },
        {
          "module": "pathlib",
          "line": 11
        },
        {
          "module": "platform",
          "line": 12
        },
        {
          "module": "shutil",
          "line": 13
        },
        {
          "module": "subprocess",
          "line": 14
        },
        {
          "module": "sys",
          "line": 15
        },
        {
          "module": "zipfile",
          "line": 16
        },
        {
          "module": "experiments.provenance",
          "line": 140
        }
      ]
    },
    {
      "path": "scripts/build_release_evidence.py",
      "sha256": "7b826a089293538db0d3cc37ca5283d1a899ad2d4f7eafb717361a94d4d69934",
      "imports": [
        {
          "module": "__future__",
          "line": 2
        },
        {
          "module": "argparse",
          "line": 4
        },
        {
          "module": "json",
          "line": 5
        },
        {
          "module": "pathlib",
          "line": 6
        },
        {
          "module": "subprocess",
          "line": 7
        },
        {
          "module": "sys",
          "line": 8
        },
        {
          "module": "experiments.common",
          "line": 12
        },
        {
          "module": "experiments.provenance",
          "line": 13
        },
        {
          "module": "experiments.verification",
          "line": 14
        }
      ]
    },
    {
      "path": "scripts/build_social_card.py",
      "sha256": "7b2762de2ad89bd49f9c4974a7945c39c0e0d0fd8bb382818a207736b92b496f",
      "imports": [
        {
          "module": "pathlib",
          "line": 2
        },
        {
          "module": "colorsys",
          "line": 3
        },
        {
          "module": "math",
          "line": 4
        },
        {
          "module": "PIL",
          "line": 6
        }
      ]
    },
    {
      "path": "scripts/check_home_animation.js",
      "sha256": "b3f6acf7347eb342c1c90c11a4670e0e3f830a0be4d63308c11439feb2f89966",
      "imports": []
    },
    {
      "path": "scripts/dashboard.py",
      "sha256": "c1d4168a02b00adc5ac3ea049a11927418cbbee7b8e41b3d60bb321e3c3c3ed1",
      "imports": [
        {
          "module": "pathlib",
          "line": 3
        },
        {
          "module": "sys",
          "line": 4
        },
        {
          "module": "dashboard.server",
          "line": 8
        }
      ]
    },
    {
      "path": "scripts/finalize_report.py",
      "sha256": "7ab674d5d85b43fd322799234cbf3730384c5fe6ad5f35866e7fe96f07ef9d5a",
      "imports": [
        {
          "module": "pathlib",
          "line": 2
        },
        {
          "module": "json",
          "line": 3
        },
        {
          "module": "os",
          "line": 4
        },
        {
          "module": "sys",
          "line": 5
        },
        {
          "module": "torch",
          "line": 10
        },
        {
          "module": "experiments.common",
          "line": 11
        },
        {
          "module": "experiments.provenance",
          "line": 12
        },
        {
          "module": "experiments.ablations",
          "line": 13
        },
        {
          "module": "experiments.instrumentation",
          "line": 14
        },
        {
          "module": "experiments.reporting",
          "line": 15
        },
        {
          "module": "experiments.precision",
          "line": 41
        },
        {
          "module": "chromatensor.visualization",
          "line": 53
        },
        {
          "module": "chromatensor",
          "line": 54
        },
        {
          "module": "chromatensor.codecs",
          "line": 55
        },
        {
          "module": "experiments.decoder",
          "line": 28
        },
        {
          "module": "experiments.attention",
          "line": 36
        }
      ]
    },
    {
      "path": "scripts/release_audit.py",
      "sha256": "4c449b86f8e24d191bb35fb27f15b59f7e8d144f85c4187c6677e3bae9bb3489",
      "imports": [
        {
          "module": "__future__",
          "line": 2
        },
        {
          "module": "argparse",
          "line": 4
        },
        {
          "module": "hashlib",
          "line": 5
        },
        {
          "module": "json",
          "line": 6
        },
        {
          "module": "pathlib",
          "line": 7
        },
        {
          "module": "re",
          "line": 8
        },
        {
          "module": "subprocess",
          "line": 9
        },
        {
          "module": "urllib.parse",
          "line": 10
        },
        {
          "module": "zipfile",
          "line": 11
        }
      ]
    },
    {
      "path": "scripts/review_reproductions.py",
      "sha256": "9f945a5abdc667727acad1edd800157cd1968030863fd9006c567cf5336a550b",
      "imports": [
        {
          "module": "__future__",
          "line": 2
        },
        {
          "module": "argparse",
          "line": 4
        },
        {
          "module": "contextlib",
          "line": 5
        },
        {
          "module": "datetime",
          "line": 6
        },
        {
          "module": "hashlib",
          "line": 7
        },
        {
          "module": "io",
          "line": 8
        },
        {
          "module": "json",
          "line": 9
        },
        {
          "module": "math",
          "line": 10
        },
        {
          "module": "pathlib",
          "line": 11
        },
        {
          "module": "subprocess",
          "line": 12
        },
        {
          "module": "sys",
          "line": 13
        },
        {
          "module": "tempfile",
          "line": 14
        },
        {
          "module": "threading",
          "line": 15
        },
        {
          "module": "urllib.request",
          "line": 16
        },
        {
          "module": "urllib.error",
          "line": 17
        },
        {
          "module": "unittest.mock",
          "line": 18
        },
        {
          "module": "tomllib",
          "line": 19
        },
        {
          "module": "zipfile",
          "line": 20
        },
        {
          "module": "torch",
          "line": 25
        },
        {
          "module": "torch",
          "line": 26
        },
        {
          "module": "chromatensor",
          "line": 27
        },
        {
          "module": "chromatensor.codecs",
          "line": 28
        },
        {
          "module": "chromatensor.autograd",
          "line": 29
        },
        {
          "module": "chromatensor.nn",
          "line": 30
        },
        {
          "module": "chromatensor.operators",
          "line": 31
        },
        {
          "module": "dashboard.server",
          "line": 32
        },
        {
          "module": "experiments.common",
          "line": 33
        },
        {
          "module": "experiments.provenance",
          "line": 34
        },
        {
          "module": "experiments.cube",
          "line": 35
        },
        {
          "module": "experiments.datasets",
          "line": 36
        },
        {
          "module": "experiments.training",
          "line": 37
        },
        {
          "module": "torchvision.datasets",
          "line": 140
        },
        {
          "module": "scripts",
          "line": 244
        }
      ]
    },
    {
      "path": "scripts/run_benchmarks.py",
      "sha256": "3c666add71610352bebffe13b09689f3977901b443db13959b0c4e3478fbd236",
      "imports": [
        {
          "module": "__future__",
          "line": 2
        },
        {
          "module": "argparse",
          "line": 4
        },
        {
          "module": "datetime",
          "line": 5
        },
        {
          "module": "json",
          "line": 6
        },
        {
          "module": "os",
          "line": 7
        },
        {
          "module": "pathlib",
          "line": 8
        },
        {
          "module": "sys",
          "line": 9
        },
        {
          "module": "torch",
          "line": 15
        },
        {
          "module": "experiments.common",
          "line": 16
        },
        {
          "module": "experiments.provenance",
          "line": 17
        },
        {
          "module": "experiments.datasets",
          "line": 18
        },
        {
          "module": "experiments.training",
          "line": 19
        },
        {
          "module": "experiments.precision",
          "line": 20
        },
        {
          "module": "experiments.reporting",
          "line": 21
        },
        {
          "module": "experiments.instrumentation",
          "line": 116
        },
        {
          "module": "chromatensor.visualization",
          "line": 120
        },
        {
          "module": "experiments.attention",
          "line": 102
        },
        {
          "module": "experiments.decoder",
          "line": 108
        },
        {
          "module": "experiments.ablations",
          "line": 114
        }
      ]
    },
    {
      "path": "scripts/run_cube_experiments.py",
      "sha256": "d65dccf54be8aaf4a2af3dc2ae73068454edc9dc9bfa428dfcfdc49fdd90ec37",
      "imports": [
        {
          "module": "pathlib",
          "line": 2
        },
        {
          "module": "datetime",
          "line": 3
        },
        {
          "module": "argparse",
          "line": 4
        },
        {
          "module": "statistics",
          "line": 5
        },
        {
          "module": "sys",
          "line": 6
        },
        {
          "module": "torch",
          "line": 10
        },
        {
          "module": "experiments.common",
          "line": 11
        },
        {
          "module": "experiments.provenance",
          "line": 12
        },
        {
          "module": "experiments.cube",
          "line": 13
        },
        {
          "module": "matplotlib",
          "line": 68
        },
        {
          "module": "matplotlib.pyplot",
          "line": 70
        }
      ]
    },
    {
      "path": "scripts/run_native_hue_package_checks.py",
      "sha256": "bd1ecf8da6fa03088ab99b254757157c2fb23482a55bc73e26fd5ae5490ff81d",
      "imports": [
        {
          "module": "__future__",
          "line": 2
        },
        {
          "module": "argparse",
          "line": 4
        },
        {
          "module": "json",
          "line": 5
        },
        {
          "module": "pathlib",
          "line": 6
        },
        {
          "module": "subprocess",
          "line": 7
        },
        {
          "module": "sys",
          "line": 8
        },
        {
          "module": "experiments.provenance",
          "line": 12
        }
      ]
    },
    {
      "path": "scripts/run_release_research.py",
      "sha256": "15cdfce4ff02543fff12df1c931982a142c57cfd9f48935a0231fc6f1f779a5b",
      "imports": [
        {
          "module": "__future__",
          "line": 3
        },
        {
          "module": "argparse",
          "line": 5
        },
        {
          "module": "datetime",
          "line": 6
        },
        {
          "module": "json",
          "line": 7
        },
        {
          "module": "math",
          "line": 8
        },
        {
          "module": "os",
          "line": 9
        },
        {
          "module": "pathlib",
          "line": 10
        },
        {
          "module": "statistics",
          "line": 11
        },
        {
          "module": "sys",
          "line": 12
        },
        {
          "module": "time",
          "line": 13
        },
        {
          "module": "numpy",
          "line": 19
        },
        {
          "module": "torch",
          "line": 20
        },
        {
          "module": "chromatensor",
          "line": 22
        },
        {
          "module": "experiments.checkpoint_probe",
          "line": 23
        },
        {
          "module": "experiments.common",
          "line": 24
        },
        {
          "module": "experiments.datasets",
          "line": 25
        },
        {
          "module": "experiments.provenance",
          "line": 26
        },
        {
          "module": "experiments.training",
          "line": 27
        },
        {
          "module": "torchvision.datasets",
          "line": 62
        },
        {
          "module": "torchvision.datasets",
          "line": 217
        }
      ]
    },
    {
      "path": "scripts/run_verification_tests.py",
      "sha256": "67cd64d256d8e08c1128f77d4f372f5b9cd23468450cd7efaaa84dc4ab669973",
      "imports": [
        {
          "module": "pathlib",
          "line": 2
        },
        {
          "module": "argparse",
          "line": 3
        },
        {
          "module": "json",
          "line": 4
        },
        {
          "module": "os",
          "line": 5
        },
        {
          "module": "subprocess",
          "line": 6
        },
        {
          "module": "sys",
          "line": 7
        },
        {
          "module": "experiments.common",
          "line": 11
        },
        {
          "module": "experiments.provenance",
          "line": 12
        }
      ]
    },
    {
      "path": "scripts/scan_release.py",
      "sha256": "2e2634e7639d75d6a73d83ac0661c77007052fe323ce1e6b074da9b68ed7235d",
      "imports": [
        {
          "module": "__future__",
          "line": 2
        },
        {
          "module": "argparse",
          "line": 4
        },
        {
          "module": "collections",
          "line": 5
        },
        {
          "module": "importlib.metadata",
          "line": 6
        },
        {
          "module": "io",
          "line": 7
        },
        {
          "module": "json",
          "line": 8
        },
        {
          "module": "pathlib",
          "line": 9
        },
        {
          "module": "re",
          "line": 10
        },
        {
          "module": "subprocess",
          "line": 11
        },
        {
          "module": "sys",
          "line": 12
        },
        {
          "module": "zipfile",
          "line": 13
        }
      ]
    },
    {
      "path": "scripts/train.py",
      "sha256": "2307323725a6dad464250e8990a3d30f4616aa97bb1d31a564cabc7af98369e7",
      "imports": [
        {
          "module": "pathlib",
          "line": 2
        },
        {
          "module": "argparse",
          "line": 3
        },
        {
          "module": "sys",
          "line": 4
        },
        {
          "module": "scripts.run_benchmarks",
          "line": 7
        }
      ]
    },
    {
      "path": "scripts/verify.py",
      "sha256": "edb0c32960052ffa22d32d029aaf2fcac17ecf290e31efe34aa96c1a8d0cd30e",
      "imports": [
        {
          "module": "pathlib",
          "line": 2
        },
        {
          "module": "datetime",
          "line": 3
        },
        {
          "module": "argparse",
          "line": 4
        },
        {
          "module": "json",
          "line": 5
        },
        {
          "module": "sys",
          "line": 6
        },
        {
          "module": "torch",
          "line": 10
        },
        {
          "module": "experiments.common",
          "line": 11
        },
        {
          "module": "experiments.provenance",
          "line": 12
        },
        {
          "module": "experiments.verification",
          "line": 13
        },
        {
          "module": "scripts.run_release_research",
          "line": 71
        },
        {
          "module": "scripts.verify_rgbd_tokens",
          "line": 81
        },
        {
          "module": "scripts.verify_native_hue_release",
          "line": 116
        }
      ]
    },
    {
      "path": "scripts/verify_dependencies.py",
      "sha256": "eda170bb50d656b72c31f1cc7de2167df79ca60baeea117231335c8406bb60e4",
      "imports": [
        {
          "module": "__future__",
          "line": 2
        },
        {
          "module": "argparse",
          "line": 4
        },
        {
          "module": "base64",
          "line": 5
        },
        {
          "module": "json",
          "line": 6
        },
        {
          "module": "pathlib",
          "line": 7
        },
        {
          "module": "shlex",
          "line": 8
        },
        {
          "module": "shutil",
          "line": 9
        },
        {
          "module": "subprocess",
          "line": 10
        },
        {
          "module": "sys",
          "line": 11
        },
        {
          "module": "xml.etree.ElementTree",
          "line": 12
        },
        {
          "module": "experiments.provenance",
          "line": 16
        },
        {
          "module": "experiments.common",
          "line": 17
        },
        {
          "module": "scripts.release_audit",
          "line": 34
        }
      ]
    },
    {
      "path": "scripts/verify_installed_hue.py",
      "sha256": "6d7d035baae3941c816c9ee03a53ef67f4b9573fe504478b924db99449fd1876",
      "imports": [
        {
          "module": "argparse",
          "line": 2
        },
        {
          "module": "hashlib",
          "line": 3
        },
        {
          "module": "importlib",
          "line": 4
        },
        {
          "module": "importlib.util",
          "line": 5
        },
        {
          "module": "json",
          "line": 6
        },
        {
          "module": "pathlib",
          "line": 7
        },
        {
          "module": "platform",
          "line": 8
        },
        {
          "module": "site",
          "line": 9
        },
        {
          "module": "sys",
          "line": 10
        },
        {
          "module": "tempfile",
          "line": 11
        },
        {
          "module": "chromatic_hue",
          "line": 23
        }
      ]
    },
    {
      "path": "scripts/verify_native_hue.py",
      "sha256": "dabad2682a51660bb68a3fffa76243572a5960f4ed79c5b2b507c3a8eaf5a648",
      "imports": [
        {
          "module": "__future__",
          "line": 2
        },
        {
          "module": "argparse",
          "line": 4
        },
        {
          "module": "array",
          "line": 5
        },
        {
          "module": "datetime",
          "line": 6
        },
        {
          "module": "hashlib",
          "line": 7
        },
        {
          "module": "json",
          "line": 8
        },
        {
          "module": "math",
          "line": 9
        },
        {
          "module": "pathlib",
          "line": 10
        },
        {
          "module": "re",
          "line": 11
        },
        {
          "module": "subprocess",
          "line": 12
        },
        {
          "module": "sys",
          "line": 13
        },
        {
          "module": "time",
          "line": 14
        },
        {
          "module": "uuid",
          "line": 15
        },
        {
          "module": "numpy",
          "line": 39
        },
        {
          "module": "numpy",
          "line": 61
        },
        {
          "module": "numpy",
          "line": 70
        },
        {
          "module": "numpy",
          "line": 89
        },
        {
          "module": "chromatensor",
          "line": 109
        },
        {
          "module": "chromatensor.hue",
          "line": 110
        },
        {
          "module": "scripts.build_native_hue",
          "line": 139
        },
        {
          "module": "numpy",
          "line": 170
        },
        {
          "module": "numpy",
          "line": 207
        },
        {
          "module": "numpy",
          "line": 239
        },
        {
          "module": "chromatensor",
          "line": 240
        },
        {
          "module": "chromatensor.hue",
          "line": 241
        },
        {
          "module": "experiments.provenance",
          "line": 242
        },
        {
          "module": "numpy",
          "line": 328
        },
        {
          "module": "chromatensor",
          "line": 329
        },
        {
          "module": "chromatensor.hue",
          "line": 330
        },
        {
          "module": "experiments.provenance",
          "line": 331
        }
      ]
    },
    {
      "path": "scripts/verify_native_hue_release.py",
      "sha256": "6117f5fdcdcb1019e7c92b9f4b8d4183498ff355da43e2a2477418c903a50b4e",
      "imports": [
        {
          "module": "__future__",
          "line": 2
        },
        {
          "module": "datetime",
          "line": 4
        },
        {
          "module": "email.parser",
          "line": 5
        },
        {
          "module": "hashlib",
          "line": 6
        },
        {
          "module": "json",
          "line": 7
        },
        {
          "module": "pathlib",
          "line": 8
        },
        {
          "module": "re",
          "line": 9
        },
        {
          "module": "subprocess",
          "line": 10
        },
        {
          "module": "sys",
          "line": 11
        },
        {
          "module": "collections",
          "line": 12
        },
        {
          "module": "xml.etree.ElementTree",
          "line": 13
        },
        {
          "module": "zipfile",
          "line": 14
        },
        {
          "module": "experiments.provenance",
          "line": 18
        },
        {
          "module": "experiments.verification",
          "line": 19
        },
        {
          "module": "scripts.build_native_hue",
          "line": 68
        },
        {
          "module": "chromatensor",
          "line": 95
        },
        {
          "module": "scripts.verify_native_hue",
          "line": 235
        },
        {
          "module": "scripts.build_native_hue",
          "line": 188
        }
      ]
    },
    {
      "path": "scripts/verify_release_install.py",
      "sha256": "55a4e4fd93fe128cec1c7c99687a0ae976016857589ff92e770a7dd7dfed498c",
      "imports": [
        {
          "module": "__future__",
          "line": 2
        },
        {
          "module": "argparse",
          "line": 4
        },
        {
          "module": "hashlib",
          "line": 5
        },
        {
          "module": "importlib",
          "line": 6
        },
        {
          "module": "json",
          "line": 7
        },
        {
          "module": "pathlib",
          "line": 8
        },
        {
          "module": "platform",
          "line": 9
        },
        {
          "module": "site",
          "line": 10
        },
        {
          "module": "sys",
          "line": 11
        },
        {
          "module": "tempfile",
          "line": 12
        },
        {
          "module": "time",
          "line": 13
        },
        {
          "module": "numpy",
          "line": 21
        },
        {
          "module": "torch",
          "line": 22
        },
        {
          "module": "chromatensor",
          "line": 23
        },
        {
          "module": "chromatensor",
          "line": 24
        },
        {
          "module": "chromatensor",
          "line": 57
        },
        {
          "module": "chromatensor.autograd",
          "line": 58
        },
        {
          "module": "experiments.checkpoint_probe",
          "line": 86
        },
        {
          "module": "experiments.common",
          "line": 87
        },
        {
          "module": "experiments.models",
          "line": 88
        }
      ]
    },
    {
      "path": "scripts/verify_rgbd_tokens.py",
      "sha256": "cfe9d167c16561c266b8af993b57334e12bea219e554e63cb0fe54381a566a69",
      "imports": [
        {
          "module": "__future__",
          "line": 2
        },
        {
          "module": "argparse",
          "line": 4
        },
        {
          "module": "datetime",
          "line": 5
        },
        {
          "module": "hashlib",
          "line": 6
        },
        {
          "module": "json",
          "line": 7
        },
        {
          "module": "math",
          "line": 8
        },
        {
          "module": "os",
          "line": 9
        },
        {
          "module": "pathlib",
          "line": 10
        },
        {
          "module": "struct",
          "line": 11
        },
        {
          "module": "subprocess",
          "line": 12
        },
        {
          "module": "sys",
          "line": 13
        },
        {
          "module": "time",
          "line": 14
        },
        {
          "module": "numpy",
          "line": 19
        },
        {
          "module": "torch",
          "line": 20
        },
        {
          "module": "chromatensor.tokenizer",
          "line": 22
        },
        {
          "module": "experiments.common",
          "line": 23
        },
        {
          "module": "experiments.provenance",
          "line": 24
        }
      ]
    },
    {
      "path": "tests/conftest.py",
      "sha256": "e1269c105af016b549e578bb980aad34e10b33424053ec32492707b70d8492b5",
      "imports": [
        {
          "module": "pathlib",
          "line": 2
        },
        {
          "module": "uuid",
          "line": 3
        },
        {
          "module": "os",
          "line": 4
        }
      ]
    },
    {
      "path": "tests/test_ablations.py",
      "sha256": "df32e8d1e261e92f5e45f92bb55fdea8410c48a3e3e16933f5cb817bbecffb10",
      "imports": [
        {
          "module": "experiments.ablations",
          "line": 1
        }
      ]
    },
    {
      "path": "tests/test_acceptance.py",
      "sha256": "82cfb263ea980b2db6c1b9d78eb1ec4ef536c5dcd78ac57200aadb5869c5121b",
      "imports": [
        {
          "module": "copy",
          "line": 2
        },
        {
          "module": "json",
          "line": 3
        },
        {
          "module": "pathlib",
          "line": 4
        },
        {
          "module": "shutil",
          "line": 5
        },
        {
          "module": "subprocess",
          "line": 6
        },
        {
          "module": "zipfile",
          "line": 7
        },
        {
          "module": "pytest",
          "line": 9
        },
        {
          "module": "experiments.provenance",
          "line": 11
        },
        {
          "module": "experiments.reporting",
          "line": 13
        },
        {
          "module": "experiments.verification",
          "line": 14
        }
      ]
    },
    {
      "path": "tests/test_architecture.py",
      "sha256": "f4d62f5fd40e201f4b22f1f58db31a51f7a893c1e3519708034ea31f1fb11fc5",
      "imports": [
        {
          "module": "__future__",
          "line": 2
        },
        {
          "module": "copy",
          "line": 4
        },
        {
          "module": "importlib.util",
          "line": 5
        },
        {
          "module": "json",
          "line": 6
        },
        {
          "module": "pathlib",
          "line": 7
        },
        {
          "module": "re",
          "line": 8
        },
        {
          "module": "pytest",
          "line": 10
        }
      ]
    },
    {
      "path": "tests/test_attention_experiment.py",
      "sha256": "8ffc19b593100388d8c8fa531f8fba8b781a2c7a2374f2da0d09b5bb79734b8f",
      "imports": [
        {
          "module": "pytest",
          "line": 1
        },
        {
          "module": "torch",
          "line": 2
        },
        {
          "module": "experiments.attention",
          "line": 4
        },
        {
          "module": "experiments.common",
          "line": 5
        }
      ]
    },
    {
      "path": "tests/test_build_native_hue.py",
      "sha256": "b9c3985e09e15ac3b11acfbd531bc0a3a32feef7e270d575cec3319e8bd25a74",
      "imports": [
        {
          "module": "json",
          "line": 2
        },
        {
          "module": "pathlib",
          "line": 3
        },
        {
          "module": "shutil",
          "line": 4
        },
        {
          "module": "subprocess",
          "line": 5
        },
        {
          "module": "zipfile",
          "line": 6
        },
        {
          "module": "pytest",
          "line": 8
        },
        {
          "module": "scripts",
          "line": 10
        },
        {
          "module": "scripts.verify_native_hue_release",
          "line": 11
        }
      ]
    },
    {
      "path": "tests/test_checkpoint_probe.py",
      "sha256": "1f595fb11f4b4df23efd1cdd18a70a638e703bff5af1e1f88da5a9e67cf63d45",
      "imports": [
        {
          "module": "copy",
          "line": 3
        },
        {
          "module": "io",
          "line": 4
        },
        {
          "module": "pathlib",
          "line": 5
        },
        {
          "module": "struct",
          "line": 6
        },
        {
          "module": "zipfile",
          "line": 7
        },
        {
          "module": "numpy",
          "line": 9
        },
        {
          "module": "pytest",
          "line": 10
        },
        {
          "module": "torch",
          "line": 11
        },
        {
          "module": "chromatensor",
          "line": 13
        },
        {
          "module": "experiments.checkpoint_probe",
          "line": 14
        },
        {
          "module": "experiments.datasets",
          "line": 18
        },
        {
          "module": "experiments.training",
          "line": 19
        },
        {
          "module": "chromatensor.codecs",
          "line": 73
        },
        {
          "module": "experiments.attention",
          "line": 90
        }
      ]
    },
    {
      "path": "tests/test_codecs.py",
      "sha256": "5b244cce5daf69e9a41e1b3f80a50cef45bef60ecba16482ff3c6a5b4b78c772",
      "imports": [
        {
          "module": "math",
          "line": 1
        },
        {
          "module": "json",
          "line": 2
        },
        {
          "module": "pytest",
          "line": 4
        },
        {
          "module": "torch",
          "line": 5
        },
        {
          "module": "chromatensor.codecs",
          "line": 7
        },
        {
          "module": "chromatensor.tokenizer",
          "line": 11
        }
      ]
    },
    {
      "path": "tests/test_cube.py",
      "sha256": "ce56c5f43379a3ddcb0c5d84938ab26bdec665021fe2aac1a494d6dbc7b15320",
      "imports": [
        {
          "module": "hashlib",
          "line": 1
        },
        {
          "module": "json",
          "line": 2
        },
        {
          "module": "zipfile",
          "line": 3
        },
        {
          "module": "pytest",
          "line": 5
        },
        {
          "module": "torch",
          "line": 6
        },
        {
          "module": "chromatensor.codecs",
          "line": 8
        },
        {
          "module": "chromatensor.cube",
          "line": 9
        },
        {
          "module": "chromatensor.tensor",
          "line": 10
        }
      ]
    },
    {
      "path": "tests/test_cube_api.py",
      "sha256": "2a1827ad175871cd8e2e4b287513b6a1a3203c6255fadf98f87e58286dcbd184",
      "imports": [
        {
          "module": "copy",
          "line": 2
        },
        {
          "module": "json",
          "line": 3
        },
        {
          "module": "math",
          "line": 4
        },
        {
          "module": "threading",
          "line": 5
        },
        {
          "module": "urllib.error",
          "line": 6
        },
        {
          "module": "urllib.request",
          "line": 7
        },
        {
          "module": "pytest",
          "line": 9
        },
        {
          "module": "dashboard.cube_api",
          "line": 11
        },
        {
          "module": "dashboard.server",
          "line": 12
        }
      ]
    },
    {
      "path": "tests/test_cube_experiments.py",
      "sha256": "39309ef4d9342d1fa9aa43de1cd31bb15af8de3ab42e92c55de99286f42b11bb",
      "imports": [
        {
          "module": "math",
          "line": 2
        },
        {
          "module": "pytest",
          "line": 4
        },
        {
          "module": "torch",
          "line": 5
        },
        {
          "module": "chromatensor.codecs",
          "line": 7
        },
        {
          "module": "chromatensor.holographic",
          "line": 8
        },
        {
          "module": "experiments.common",
          "line": 9
        },
        {
          "module": "experiments.cube",
          "line": 10
        },
        {
          "module": "json",
          "line": 105
        },
        {
          "module": "scripts",
          "line": 106
        },
        {
          "module": "experiments.common",
          "line": 134
        },
        {
          "module": "experiments.cube",
          "line": 135
        },
        {
          "module": "chromatensor",
          "line": 136
        },
        {
          "module": "chromatensor.nn",
          "line": 137
        },
        {
          "module": "experiments.models",
          "line": 138
        },
        {
          "module": "experiments.cube",
          "line": 151
        },
        {
          "module": "experiments.provenance",
          "line": 152
        }
      ]
    },
    {
      "path": "tests/test_dashboard.py",
      "sha256": "76608e25cb244b7c5f935859d5a7cf8752729bffa9962a3a6c22deb8572bf966",
      "imports": [
        {
          "module": "json",
          "line": 3
        },
        {
          "module": "http.client",
          "line": 4
        },
        {
          "module": "socket",
          "line": 5
        },
        {
          "module": "threading",
          "line": 6
        },
        {
          "module": "urllib.error",
          "line": 7
        },
        {
          "module": "urllib.parse",
          "line": 8
        },
        {
          "module": "urllib.request",
          "line": 9
        },
        {
          "module": "pytest",
          "line": 11
        },
        {
          "module": "dashboard.server",
          "line": 13
        }
      ]
    },
    {
      "path": "tests/test_datasets.py",
      "sha256": "aeb1c8fbe0a16a4f348bcbb28923e0bce3b958b42b953c57c70ca5eceeaa1331",
      "imports": [
        {
          "module": "pytest",
          "line": 1
        },
        {
          "module": "torch",
          "line": 2
        },
        {
          "module": "experiments.datasets",
          "line": 4
        }
      ]
    },
    {
      "path": "tests/test_decoder.py",
      "sha256": "c1e8a7b19cde5ee2b7da8f1d288c578f47198e4ce7e5c2adef432f1375cf1a92",
      "imports": [
        {
          "module": "json",
          "line": 1
        },
        {
          "module": "pytest",
          "line": 3
        },
        {
          "module": "torch",
          "line": 4
        },
        {
          "module": "chromatensor.codecs",
          "line": 6
        },
        {
          "module": "chromatensor.decoders",
          "line": 7
        },
        {
          "module": "chromatensor.tensor",
          "line": 8
        }
      ]
    },
    {
      "path": "tests/test_dependency_runner.py",
      "sha256": "4b925e88fea594cb7528487f39c931d2d9ab57e5f170d770d65841265be4b1cc",
      "imports": [
        {
          "module": "pathlib",
          "line": 2
        },
        {
          "module": "pytest",
          "line": 4
        },
        {
          "module": "scripts",
          "line": 6
        }
      ]
    },
    {
      "path": "tests/test_file_limits.py",
      "sha256": "bf1ded7fb665bbabf7f8782f3770babeb2e3553ed8fa1a6a8b1826d36acbe1d5",
      "imports": [
        {
          "module": "hashlib",
          "line": 2
        },
        {
          "module": "io",
          "line": 3
        },
        {
          "module": "json",
          "line": 4
        },
        {
          "module": "struct",
          "line": 5
        },
        {
          "module": "zipfile",
          "line": 6
        },
        {
          "module": "zlib",
          "line": 7
        },
        {
          "module": "numpy",
          "line": 9
        },
        {
          "module": "pytest",
          "line": 10
        },
        {
          "module": "torch",
          "line": 11
        },
        {
          "module": "chromatensor",
          "line": 13
        },
        {
          "module": "chromatensor.holographic",
          "line": 17
        },
        {
          "module": "chromatensor.serialization",
          "line": 18
        }
      ]
    },
    {
      "path": "tests/test_holographic.py",
      "sha256": "b448ab2e0f04b18855b976f80bdf3fc382deb4747faa541c6de17b74f20a917f",
      "imports": [
        {
          "module": "hashlib",
          "line": 1
        },
        {
          "module": "json",
          "line": 2
        },
        {
          "module": "math",
          "line": 3
        },
        {
          "module": "struct",
          "line": 4
        },
        {
          "module": "zlib",
          "line": 5
        },
        {
          "module": "pytest",
          "line": 7
        },
        {
          "module": "torch",
          "line": 8
        },
        {
          "module": "chromatensor.holographic",
          "line": 10
        },
        {
          "module": "chromatensor.holographic",
          "line": 11
        }
      ]
    },
    {
      "path": "tests/test_hue_packaging.py",
      "sha256": "89900b6cdbd5224a378dad5dc5e55fcd500377efaa956ef0914203e131c9d9d0",
      "imports": [
        {
          "module": "types",
          "line": 2
        },
        {
          "module": "pytest",
          "line": 4
        },
        {
          "module": "chromatensor",
          "line": 6
        },
        {
          "module": "scripts",
          "line": 7
        },
        {
          "module": "scripts.verify_installed_hue",
          "line": 8
        }
      ]
    },
    {
      "path": "tests/test_instrumentation.py",
      "sha256": "acd4232b4c6db004494e8471b0aabe86d626fa731a7f3342cfbc572a83347bb3",
      "imports": [
        {
          "module": "pathlib",
          "line": 1
        },
        {
          "module": "numpy",
          "line": 3
        },
        {
          "module": "pytest",
          "line": 4
        },
        {
          "module": "torch",
          "line": 5
        },
        {
          "module": "chromatensor",
          "line": 7
        },
        {
          "module": "chromatensor.nn",
          "line": 8
        },
        {
          "module": "experiments.instrumentation",
          "line": 9
        },
        {
          "module": "experiments.training",
          "line": 10
        },
        {
          "module": "chromatensor.codecs",
          "line": 88
        }
      ]
    },
    {
      "path": "tests/test_native_hue.py",
      "sha256": "58d60cc357bf4734cc83a7a91d39b161919643229bc465433eeef0748e1ed8dd",
      "imports": [
        {
          "module": "__future__",
          "line": 2
        },
        {
          "module": "copy",
          "line": 4
        },
        {
          "module": "fractions",
          "line": 5
        },
        {
          "module": "hashlib",
          "line": 6
        },
        {
          "module": "json",
          "line": 7
        },
        {
          "module": "pathlib",
          "line": 8
        },
        {
          "module": "struct",
          "line": 9
        },
        {
          "module": "subprocess",
          "line": 10
        },
        {
          "module": "sys",
          "line": 11
        },
        {
          "module": "numpy",
          "line": 13
        },
        {
          "module": "pytest",
          "line": 14
        },
        {
          "module": "chromatensor",
          "line": 16
        },
        {
          "module": "chromatensor.hue",
          "line": 17
        },
        {
          "module": "scripts",
          "line": 18
        },
        {
          "module": "experiments.provenance",
          "line": 371
        }
      ]
    },
    {
      "path": "tests/test_native_release.py",
      "sha256": "a9361b1648082c4294124683974016a2d2602635c38baac6a8d56047f51797d9",
      "imports": [
        {
          "module": "copy",
          "line": 2
        },
        {
          "module": "pytest",
          "line": 4
        },
        {
          "module": "experiments.provenance",
          "line": 6
        },
        {
          "module": "scripts.verify_native_hue_release",
          "line": 7
        }
      ]
    },
    {
      "path": "tests/test_nn.py",
      "sha256": "d3908175692e1e973cd7c113fa151a3a1bb701e8c322383e511748994dae770c",
      "imports": [
        {
          "module": "math",
          "line": 3
        },
        {
          "module": "pytest",
          "line": 5
        },
        {
          "module": "torch",
          "line": 6
        },
        {
          "module": "torch",
          "line": 7
        },
        {
          "module": "chromatensor.autograd",
          "line": 9
        },
        {
          "module": "chromatensor.nn",
          "line": 10
        },
        {
          "module": "chromatensor.nn.layers",
          "line": 218
        }
      ]
    },
    {
      "path": "tests/test_operators.py",
      "sha256": "990203f49d2e13bca02dd7f4455aa28e68baafe39aa63ab374b869246f7e45c3",
      "imports": [
        {
          "module": "pytest",
          "line": 1
        },
        {
          "module": "torch",
          "line": 2
        },
        {
          "module": "chromatensor",
          "line": 4
        },
        {
          "module": "chromatensor.operators",
          "line": 5
        }
      ]
    },
    {
      "path": "tests/test_public_release_manifest.py",
      "sha256": "09fb7fd27045b7c17c265a56b0487c9be222c81c706d36303c92a60c5dc7abca",
      "imports": [
        {
          "module": "json",
          "line": 2
        },
        {
          "module": "pathlib",
          "line": 3
        },
        {
          "module": "zipfile",
          "line": 4
        },
        {
          "module": "pytest",
          "line": 6
        },
        {
          "module": "scripts.release_audit",
          "line": 8
        }
      ]
    },
    {
      "path": "tests/test_release_matrix.py",
      "sha256": "54edcc832000ccda014dbfe0ab0a349cfef165a6a127a4a6aeeac41fa1a4a014",
      "imports": [
        {
          "module": "pytest",
          "line": 1
        },
        {
          "module": "scripts.verify_dependencies",
          "line": 3
        }
      ]
    },
    {
      "path": "tests/test_release_research.py",
      "sha256": "afa406fab27e92708eb1f5912a3e06ef942d3f27ef7b0ab540d190770c71108f",
      "imports": [
        {
          "module": "copy",
          "line": 3
        },
        {
          "module": "pathlib",
          "line": 4
        },
        {
          "module": "numpy",
          "line": 6
        },
        {
          "module": "pytest",
          "line": 7
        },
        {
          "module": "torch",
          "line": 8
        },
        {
          "module": "experiments.common",
          "line": 10
        },
        {
          "module": "scripts",
          "line": 11
        }
      ]
    },
    {
      "path": "tests/test_release_scan.py",
      "sha256": "501eb7611d32db3d84ac0e4433cc5fdfc19c0d295c565109d67b375b0e59e8e1",
      "imports": [
        {
          "module": "importlib.util",
          "line": 2
        },
        {
          "module": "io",
          "line": 3
        },
        {
          "module": "json",
          "line": 4
        },
        {
          "module": "pathlib",
          "line": 5
        },
        {
          "module": "zipfile",
          "line": 6
        }
      ]
    },
    {
      "path": "tests/test_result_io.py",
      "sha256": "58e4a2b9d42b9b338729aeea75e7d029825252d654166549c838a891e3868bee",
      "imports": [
        {
          "module": "json",
          "line": 1
        },
        {
          "module": "os",
          "line": 2
        },
        {
          "module": "experiments.common",
          "line": 3
        }
      ]
    },
    {
      "path": "tests/test_rgbd_nn.py",
      "sha256": "a09d41ece470bed998f29acdf50931f91224b63401166d7746a81997211c0aa1",
      "imports": [
        {
          "module": "struct",
          "line": 3
        },
        {
          "module": "pytest",
          "line": 5
        },
        {
          "module": "torch",
          "line": 6
        },
        {
          "module": "torch",
          "line": 7
        },
        {
          "module": "chromatensor.autograd",
          "line": 9
        },
        {
          "module": "chromatensor.codecs",
          "line": 10
        },
        {
          "module": "chromatensor.nn",
          "line": 11
        }
      ]
    },
    {
      "path": "tests/test_rgbd_tokens.py",
      "sha256": "c219392c56391983f02da4ccb0e0e1053d8cd244af61af790b3d9183dee6b94f",
      "imports": [
        {
          "module": "hashlib",
          "line": 2
        },
        {
          "module": "copy",
          "line": 3
        },
        {
          "module": "json",
          "line": 4
        },
        {
          "module": "pathlib",
          "line": 5
        },
        {
          "module": "struct",
          "line": 6
        },
        {
          "module": "subprocess",
          "line": 7
        },
        {
          "module": "sys",
          "line": 8
        },
        {
          "module": "numpy",
          "line": 10
        },
        {
          "module": "pytest",
          "line": 11
        },
        {
          "module": "torch",
          "line": 12
        },
        {
          "module": "chromatensor.codecs",
          "line": 14
        },
        {
          "module": "chromatensor.tokenizer",
          "line": 15
        },
        {
          "module": "argparse",
          "line": 286
        },
        {
          "module": "scripts.verify_rgbd_tokens",
          "line": 287
        },
        {
          "module": "scripts.verify_rgbd_tokens",
          "line": 294
        },
        {
          "module": "scripts.verify_rgbd_tokens",
          "line": 304
        },
        {
          "module": "scripts.verify_rgbd_tokens",
          "line": 313
        },
        {
          "module": "experiments.provenance",
          "line": 348
        },
        {
          "module": "scripts.verify_rgbd_tokens",
          "line": 349
        }
      ]
    },
    {
      "path": "tests/test_serialization.py",
      "sha256": "1a33cf704b47831070438c3d7d99ad5c268ccab3039ca73b43ed2a46cdc4ccd7",
      "imports": [
        {
          "module": "hashlib",
          "line": 1
        },
        {
          "module": "json",
          "line": 2
        },
        {
          "module": "struct",
          "line": 3
        },
        {
          "module": "pytest",
          "line": 5
        },
        {
          "module": "torch",
          "line": 6
        },
        {
          "module": "chromatensor",
          "line": 8
        }
      ]
    },
    {
      "path": "tests/test_tensor.py",
      "sha256": "cdaffb1041bf3d47643b9a5c3dfe60962ea7624862094f333be0484f68526d90",
      "imports": [
        {
          "module": "pytest",
          "line": 1
        },
        {
          "module": "numpy",
          "line": 2
        },
        {
          "module": "torch",
          "line": 3
        },
        {
          "module": "chromatensor.tensor",
          "line": 5
        },
        {
          "module": "chromatensor.codecs",
          "line": 6
        }
      ]
    },
    {
      "path": "tests/test_training.py",
      "sha256": "e159375ee9c0605212a1ab5043ed8497607ab8176ce81f588017ac426777c50b",
      "imports": [
        {
          "module": "os",
          "line": 3
        },
        {
          "module": "pathlib",
          "line": 4
        },
        {
          "module": "pytest",
          "line": 8
        },
        {
          "module": "torch",
          "line": 9
        },
        {
          "module": "experiments.datasets",
          "line": 11
        },
        {
          "module": "experiments.models",
          "line": 12
        },
        {
          "module": "experiments.training",
          "line": 13
        },
        {
          "module": "chromatensor.nn",
          "line": 14
        },
        {
          "module": "experiments.common",
          "line": 124
        },
        {
          "module": "experiments.training",
          "line": 125
        }
      ]
    }
  ],
  "source_identity": "760e25da5d324dba6af03fb78f2571f2c7405aa13a586943009c71f5b998493f",
  "evidence_note": "Source links resolve to generated snapshots of actual local files. Hash and symbol checks detect drift. Tests listed here identify relevant verification code, not a claim that those tests passed in this run. Semantic edges are reviewed assertions, separate from lexical import discovery. Absent integrations describe this reviewed source, not future plans.",
  "semantic_review": {
    "method": "Manually curated source behavior and ownership contracts, with exact source anchors. Validation protects the reviewed registry from stale or altered graph data; it does not automatically prove program behavior.",
    "registry": "scripts/architecture.py",
    "automatic_discovery_creates_edges": false
  },
  "status_note": "Implemented means the reviewed source has a runnable implementation, not that current release gates passed. Experimental marks numerical research APIs and experiments. Absent means no implementation or integration was found in the reviewed runtime. This map has no event bus or streaming edge because no such runtime path was found."
};
