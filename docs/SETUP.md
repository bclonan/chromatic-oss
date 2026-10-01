# Setup

Use [the developer guide](DEVELOPER_GUIDE.md) for native-only and full research setup. Use [the release guide](RELEASE_GUIDE.md) to build and install the current wheels. The [testing guide](TESTING.md) distinguishes host, isolated CPU, native-only and Linux evidence.

The original procedure assumed pre-existing wheels, fixed local environments and historical receipts. Build the native wheel from the included Rust source. Research wheels remain portable Python packages and require the separate native wheel for installed native aliases.

No cloud services, credentials or database migrations are required. Keep the dashboard on loopback. [Configuration](CONFIGURATION.md) lists command-line settings and numerical experiment environment conventions.
