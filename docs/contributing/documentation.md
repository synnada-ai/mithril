# Documentation Guide

This guide outlines the standards and procedures for creating and maintaining documentation for the Mithril project.

## Documentation Philosophy

Good documentation is crucial for Mithril's usability and adoption. Our documentation aims to be:

- **Comprehensive**: Cover all important features and concepts
- **Clear**: Use simple language and avoid jargon
- **Structured**: Organized in a logical, navigable manner
- **Current**: Updated alongside code changes
- **Practical**: Include examples and use cases

## Documentation Structure

The Mithril documentation is structured as follows:

- **Getting Started**: Installation, quick start guides, and core concepts
- **User Guide**: Detailed explanations of features and components
- **API Reference**: Comprehensive API documentation
- **Examples**: Real-world usage examples
- **Tutorials**: Step-by-step guides for specific tasks
- **Contributing**: Guides for contributors
- **Benchmarks**: Performance benchmarks
- **Changelog**: Version history and changes

## Documentation Sources

Documentation comes from these sources:

1. **Markdown Documentation**: Files in the `docs/` directory
2. **Docstrings**: Python docstrings in the source code
3. **Examples**: Example code in the `examples/` directory
4. **Tutorials**: Tutorial code and explanations

## Markdown Documentation

Most of Mithril's documentation is written in Markdown files in the `docs/` directory.

### File Organization

- Each major section has its own directory (`getting-started/`, `user-guide/`, etc.)
- Files should have descriptive names (`installation.md`, `model-composition.md`)
- Use consistent naming conventions (kebab-case)
- Structure follows the navigation defined in `mkdocs.yml`

### Style Guidelines

#### Headers

- Use title case for headers (capitalize major words)
- Use ATX-style headers with a space after the hash (`# Title` not `#Title`)
- Nest headers logically (H1 → H2 → H3)
- Limit the use of H1 to one per file (the title)

#### Text Formatting

- Use **bold** for emphasis
- Use *italics* for terminology or parameters
- Use `code` for code snippets, function names, class names, etc.
- Use bullet points and numbered lists for clear organization
- Keep paragraphs relatively short (3-5 sentences)

#### Code Blocks

- Use fenced code blocks with language specified
- Include meaningful examples
- Add comments where helpful

```python
# Create a simple model
model = Model()
model |= Linear(dimension=64)(input="input", output="hidden")
model += Relu()(input="hidden", output="hidden_act")
model += Linear(dimension=10)(input="hidden_act", output="output")
```

#### Links

- Use relative links for internal documentation
- Use descriptive link text

```markdown
See the [Model Composition](../user-guide/model-composition.md) guide for more information.
```

### Images and Diagrams

- Place images in the `docs/assets/` directory
- Use descriptive filenames
- Include alt text for accessibility
- Use SVG format for diagrams when possible
- Keep image file sizes reasonable for web viewing

## API Documentation

API documentation comes from docstrings in the code.

### Docstring Style

Mithril uses NumPy-style docstrings:

```python
def compile_model(
    model: Model, 
    backend: Backend, 
    shapes: Optional[Dict[str, List[int]]] = None
) -> CompiledModel:
    """
    Compile a logical model to a physical model for the specified backend.
    
    Parameters
    ----------
    model : Model
        The logical model to compile
    backend : Backend
        The backend to compile for
    shapes : Dict[str, List[int]], optional
        Dictionary mapping input names to their shapes
        
    Returns
    -------
    CompiledModel
        The compiled model ready for execution
        
    Notes
    -----
    Compilation involves shape inference, type inference, and backend-specific
    optimizations. For complex models, this may take some time.
    
    Examples
    --------
    >>> model = Model()
    >>> model |= Linear(dimension=10)(input="input", output="output")
    >>> compiled_model = compile_model(model, JaxBackend())
    >>> outputs = compiled_model.evaluate({"input": np.random.randn(1, 5)})
    """
```

### Component Documentation

For all modules, classes, functions, and methods:

- Document the purpose and behavior
- Document parameters, return values, and exceptions
- Include type annotations
- Provide examples where helpful
- Note any performance considerations or warnings

## Example Code

Example code should:

- Be complete and runnable
- Demonstrate realistic use cases
- Include comments explaining key steps
- Be simple enough for beginners to understand
- Demonstrate best practices

## Tutorial Documentation

Tutorials should:

- Walk through a complete task step by step
- Explain the reasoning behind each step
- Start from basics and build progressively
- Include all code needed to run the example
- Link to relevant API documentation

## Writing Documentation

### Creating New Documentation

1. Identify what needs documentation
2. Determine the appropriate location in the documentation structure
3. Create the file following the style guidelines
4. Update `mkdocs.yml` to include the new file in navigation
5. Build the documentation locally to verify

### Updating Documentation

When making code changes:

1. Update relevant docstrings
2. Update any affected user guide pages
3. Update examples if APIs have changed
4. Add to the changelog if appropriate

## Building Documentation

Mithril uses MkDocs with the Material theme for documentation.

### Local Build

To build and serve documentation locally:

```bash
# Install dependencies
pip install -r requirements/dev.txt

# Serve documentation with live reloading
mkdocs serve
```

View the documentation at `http://127.0.0.1:8000/`.

### Build Checks

Before submitting documentation changes:

1. Check for warnings during the build
2. Verify all links work correctly
3. Ensure all code examples run correctly
4. Check rendering of all pages

## Documentation Review Process

Documentation changes follow the same review process as code:

1. Create a branch for your changes
2. Make the changes
3. Submit a pull request
4. Address review feedback

## Documentation Maintenance

Regular maintenance tasks:

- Review and update documentation for each release
- Fix broken links and examples
- Update screenshots and diagrams as needed
- Keep installation instructions current
- Update compatibility information

## Common Documentation Issues

### Unclear Explanations

- Identify the target audience
- Define terms before using them
- Use concrete examples
- Break complex concepts into simpler parts

### Missing Information

- Cover edge cases and limitations
- Include error handling
- Provide troubleshooting tips
- Link to related documentation

### Outdated Documentation

- Check documentation during release process
- Use tests to verify examples
- Encourage users to report documentation issues

## Documentation Tools

Mithril documentation uses:

- **MkDocs**: Documentation generator
- **Material for MkDocs**: Theme for MkDocs
- **Markdown**: Content format
- **PyMdown Extensions**: Markdown extensions for enhanced features
- **GitHub**: Version control for documentation

## Resources for Documentation Writers

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [Markdown Guide](https://www.markdownguide.org/)
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html)
- [Google Technical Writing Courses](https://developers.google.com/tech-writing)