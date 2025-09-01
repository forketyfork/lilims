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
            
            # Just command runner
            just
          ];

          shellHook = ''
            echo "🚀 Lilims development environment"
            
            echo "Swift version: $(swift --version | head -n1)"
            echo "Python version: $(python --version)"
            echo "Ruff version: $(ruff --version)"
            
            
            echo "📋 Available commands (via just):"
            echo "  just clean     - Clean build artifacts"
            echo "  just build     - Build the project"
            echo "  just test      - Run all tests (Swift and Python)"
            echo "  just lint      - Run linting (Python and Swift tests)"
          '';

       };

        # Formatter for nix files
        formatter = pkgs.nixpkgs-fmt;
      });
}
