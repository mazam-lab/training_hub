# Developing Training Hub

This guide covers the development workflow for Training Hub, including setup, testing, and documentation.

## Development Setup

### Installing in Editable Mode

Install Training Hub in editable mode to test changes without reinstalling:

```bash
# Clone the repository
git clone https://github.com/Red-Hat-AI-Innovation-Team/training_hub.git
cd training_hub

# Install in editable mode
pip install -e .
```

### Installing with Optional Dependencies

Training Hub supports different CUDA versions and optional features. Install the appropriate extras for your environment:

```bash
# CUDA 12.1 support (default)
pip install -e ".[cuda]"

# CUDA 11.8 support
pip install -e ".[cuda118]"

# Development dependencies (testing, linting, etc.)
pip install -e ".[dev]"

# All dependencies
pip install -e ".[cuda,dev]"
```

### Installing Backend Dependencies

Training Hub uses pluggable backends. Install the backends you need:

```bash
# InstructLab Training backend (for SFT)
pip install instructlab-training

# Mini-Trainer backend (for OSFT)
pip install rhai-innovation-mini-trainer

# Optional: Liger kernels for performance
pip install liger-kernel
```

## Documentation Development

Training Hub uses [Docsify](https://docsify.js.org/) for documentation. You can preview documentation changes locally before committing.

### Installing Docsify

Install Docsify CLI globally:

```bash
npm install -g docsify-cli
```

### Running Local Documentation Server

Start a local documentation server:

```bash
# From the repository root
cd docs
docsify serve

# Or specify the docs directory
docsify serve docs
```

The documentation will be available at `http://localhost:3000`.

The server will automatically reload when you make changes to markdown files.

### Documentation Structure

```
docs/
├── _sidebar.md          # Sidebar navigation
├── _navbar.md           # Top navigation bar
├── _coverpage.md        # Landing page cover
├── README.md            # Home page
├── index.html           # Docsify configuration
├── algorithms/          # Algorithm overviews
├── api/                 # API reference
│   ├── functions/       # Function documentation
│   ├── classes/         # Class documentation
│   └── backends/        # Backend documentation
├── guides/              # How-to guides
└── examples/            # Examples overview
```

### Documentation Guidelines

1. **Use absolute paths** for internal links: `/api/functions/sft` instead of `../functions/sft.md`
2. **Link to GitHub** for source references: Use simple GitHub blob links instead of file paths with line numbers
3. **Keep it minimal**: Start with TBD placeholders for sections that may not be needed
4. **Test locally**: Always preview documentation changes with `docsify serve` before committing

## Development Workflow

### Making Code Changes

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes with editable install active:
   ```bash
   # Your changes are immediately available
   python -c "from training_hub import sft; print(sft)"
   ```

3. Test your changes:
   ```bash
   # Run tests (if available)
   pytest tests/

   # Or manually test
   python examples/scripts/your_test_script.py
   ```

### Making Documentation Changes

1. Edit markdown files in the `docs/` directory

2. Preview changes locally:
   ```bash
   cd docs
   docsify serve
   ```

3. Verify all links work correctly

4. Check that sidebar navigation is updated if you added new pages

### Code Style

Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for all Python code.

## Testing

### Manual Testing

Test your changes with real training runs:

```bash
# Test SFT
python examples/scripts/sft_qwen2.5_7b.py

# Test OSFT
python examples/scripts/osft_qwen2.5_7b.py
```

### Running Examples

All examples in `examples/` should work with your development installation:

```bash
# Jupyter notebooks
cd examples/notebooks
jupyter notebook

# Python scripts
cd examples/scripts
python sft_qwen2.5_7b.py --help
```

## Common Development Tasks

### Adding a New Algorithm

1. Create algorithm class in `src/training_hub/algorithms/`
2. Register in `AlgorithmRegistry`
3. Add convenience function wrapper
4. Create documentation in `docs/algorithms/`
5. Add API reference in `docs/api/`
6. Add examples in `examples/`

### Adding a New Backend

1. Create backend class inheriting from `Backend`
2. Implement `execute_training()` method
3. Register with `AlgorithmRegistry`
4. Add documentation in `docs/api/backends/`
5. Test with existing algorithms

### Updating Documentation

1. Edit relevant `.md` files in `docs/`
2. Update `docs/_sidebar.md` if adding new pages
3. Preview with `docsify serve`
4. Verify all links use absolute paths
5. Check GitHub source links are correct

## Troubleshooting Development Issues

### Editable Install Not Working

If changes aren't reflected:

```bash
# Reinstall in editable mode
pip uninstall training-hub
pip install -e .

# Or force reinstall
pip install -e . --force-reinstall --no-deps
```

### Docsify Not Found

Install Docsify globally:

```bash
npm install -g docsify-cli

# Or use npx (doesn't require global install)
npx docsify-cli serve docs
```

### Import Errors with Backends

Ensure backend packages are installed:

```bash
pip install instructlab-training rhai-innovation-mini-trainer
```

## Contributing

When you're ready to contribute:

1. Ensure your code follows the style guide
2. Test your changes thoroughly
3. Update documentation as needed
4. Create a pull request with a clear description

For more information, see the [Extending the Framework Guide](/guides/extending-framework).

## Resources

- [**Docsify Documentation**](https://docsify.js.org/) - Documentation framework
- [**Google Python Style Guide**](https://google.github.io/styleguide/pyguide.html) - Code style
- [**GitHub Repository**](https://github.com/Red-Hat-AI-Innovation-Team/training_hub) - Source code
- [**Extending the Framework**](/guides/extending-framework) - Creating custom algorithms and backends
