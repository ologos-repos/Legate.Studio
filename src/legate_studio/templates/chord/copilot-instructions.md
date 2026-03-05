# Copilot Instructions

This project was created by Legate Studio. Follow these guidelines when implementing.

## Project Context

- **Name**: {{ project_name }}
- **Type**: Chord (multi-phase project)
- **Intent**: {{ project_description }}

## Development Guidelines

### Code Style
- Write clean, readable code with clear variable names
- Add docstrings to functions and classes
- Keep functions focused and small
- Use type hints where appropriate

### Testing
- Write tests for new functionality in `tests/`
- Ensure tests pass before completing work
- Aim for meaningful test coverage, not just line coverage

### Documentation
- Update README.md if adding new features
- Document architectural decisions in `docs/`
- Keep SIGNAL.md updated with project status

### Git Practices
- Write clear, descriptive commit messages
- Keep commits focused on single changes
- Reference issue numbers in commits

## Phase Implementation

This is a multi-phase project. Check `/plans` for the current phase requirements.

1. **Phase 1: Foundation** - Set up project structure, dependencies, basic scaffolding
2. **Phase 2: Core** - Implement main functionality
3. **Phase 3: Integration** - Connect components, add polish, finalize

## Constraints

- Follow existing patterns in the codebase
- Don't introduce unnecessary dependencies
- Keep the solution simple and maintainable
- Ask for clarification if requirements are unclear

## References

- [SIGNAL.md](./SIGNAL.md) - Project intent and context
- [docs/architecture.md](./docs/architecture.md) - System design
- `/plans/` - Phase-specific implementation details
