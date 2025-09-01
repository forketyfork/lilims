{
  description = "Lilims - Swift Core ML inference runtime for iOS";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Python 3.12 (for coremltools compatibility)
            python312

            # Python development
            ruff
            
            # Development utilities
            git
            curl
            jq
            
            # File utilities mentioned in CLAUDE.md
            ripgrep  # rg
            fd       # fd
            tree
            
            # GitHub CLI
            gh
            
            # Just command runner
            just
           
           ];

          shellHook = ''
            # Use system Xcode instead of Nix SDK
            unset DEVELOPER_DIR SDKROOT
            
            # Put system binaries first in PATH to override Nix Swift
            export PATH="/usr/bin:$PATH"

            echo "🚀 Lilims development environment"
            echo "🔧 Using system Xcode from: $(xcode-select -p 2>/dev/null || echo 'Xcode not found')"
            echo "🔧 Using system Swift from: $(which swift 2>/dev/null || echo 'Swift not found')"
            
            if command -v swift >/dev/null 2>&1; then
              echo "Swift version: $(swift --version | head -n1)"
            fi
            echo "Python version: $(python --version)"
            echo "Ruff version: $(ruff --version)"
            
            echo "📋 Available commands (via just):"
            echo "  just clean     - Clean build artifacts"
            echo "  just build     - Build the project (using xcodebuild)"
            echo "  just test      - Run all tests (using xcodebuild + Python)"
            echo "  just lint      - Run linting (Python and Swift tests)"
          '';

       };

        # Formatter for nix files
        formatter = pkgs.nixpkgs-fmt;
      });
}
