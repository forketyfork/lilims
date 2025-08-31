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
        
        # Python environment with project dependencies
        pythonEnv = pkgs.python313.withPackages (ps: with ps; [
          numpy
          pytest
        ]);

      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Swift development
            swift
            swiftformat
            swiftlint
            
            # Apple development tools (macOS only)
          ] ++ lib.optionals stdenv.isDarwin [
            # macOS development tools available when on Darwin
          ] ++ [
            # Python development
            pythonEnv
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
          ];

          shellHook = ''
            echo "🚀 Lilims development environment"
            echo "Swift version: $(swift --version | head -n1)"
            echo "Python version: $(python --version)"
            echo "Ruff version: $(ruff --version)"
            
            # Swift package manager cache
            export SWIFTPM_CACHE_DIR="$PWD/.build/cache"
            
            # Helpful aliases
            alias test-swift="swift test"
            alias test-python="pytest Scripts/tests"
            alias lint-python="ruff check Scripts"
            alias lint-all="ruff check Scripts && swift test"
            
            echo "📋 Available commands:"
            echo "  test-swift     - Run Swift tests"
            echo "  test-python    - Run Python tests" 
            echo "  lint-python    - Lint Python code"
            echo "  lint-all       - Run all linting and tests"
          '';

          # Environment variables
          NIX_ENFORCE_PURITY = 0; # Allow impure operations needed for Swift/Xcode
        };

        # Formatter for nix files
        formatter = pkgs.nixpkgs-fmt;
      });
}