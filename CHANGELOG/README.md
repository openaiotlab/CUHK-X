# CHANGELOG Folder Guide

This folder contains templates and logs to help track all changes in the CUHK-X project systematically.

## 📁 File Structure

```
CHANGELOG/
├── README.md          # This guide
├── CHANGELOG.md       # Main project changelog (public releases)
├── CODELOG.md         # Code-level changes (features, bugs, refactoring)
├── DATALOG.md         # Dataset changes (new data, schema updates, corrections)
└── HARDWARE.md        # Hardware & infrastructure changes
```

## 📋 When to Use Each File

### CHANGELOG.md
**Purpose**: Public-facing release notes for version announcements  
**Use when**:
- Publishing a new version (v1.x.x)
- Major milestones or releases
- Summarizing changes for external users
- Combining multiple changes into a release

**Key sections**:
- Version number and date
- Highlights (brief summary)
- Added/Changed/Fixed/Deprecated
- Breaking changes and migration notes
- Performance metrics and baseline updates

### CODELOG.md
**Purpose**: Detailed code-level change tracking for developers  
**Use when**:
- Adding new features or functions
- Fixing bugs
- Refactoring code
- Optimizing performance
- Updating APIs or dependencies

**Key sections**:
- Component/file path
- Type of change (Feature/Bugfix/Refactor/etc.)
- Detailed changes by file
- Impact on performance and APIs
- Testing coverage

### DATALOG.md
**Purpose**: Dataset changes and data quality tracking  
**Use when**:
- Adding new data samples
- Updating data schema or metadata
- Fixing data quality issues
- Correcting annotations
- Changing data splits
- Releasing new dataset versions

**Key sections**:
- Dataset version (semantic versioning)
- Data statistics (before/after)
- Schema updates
- Quality improvements
- Backward compatibility notes
- Checksums and download links

### HARDWARE.md
**Purpose**: Hardware, infrastructure, and environment tracking  
**Use when**:
- Adding/upgrading hardware (GPUs, servers)
- Updating system requirements
- Changing dependencies or software stack
- Deploying infrastructure changes
- Documenting compatibility issues
- Setting up new environments

**Key sections**:
- Hardware specifications
- Software environment (OS, CUDA, Python, etc.)
- Dependencies changes
- Performance impact
- Known issues and workarounds
- Compatibility matrix

## 🎯 Quick Start

### 1. Choose the Right File
Ask yourself:
- Is this a public release? → **CHANGELOG.md**
- Did I modify code? → **CODELOG.md**
- Did I change data or annotations? → **DATALOG.md**
- Did I update hardware/environment? → **HARDWARE.md**

### 2. Copy the Template
Each file has a template section at the top. Copy it and fill in your changes.

### 3. Fill in Details
Be specific and thorough:
- ✅ "Added `calculate_accuracy()` function to `eval.py` with support for top-k metrics"
- ❌ "Updated eval code"

### 4. Include Context
- **Why**: Why was this change needed?
- **What**: What exactly changed?
- **Impact**: How does it affect users/performance/compatibility?
- **Testing**: How was it verified?

### 5. Add Dates and Authors
Always include:
- Date in YYYY-MM-DD format
- Your name or team name
- Version numbers (for releases and data)

## 💡 Best Practices

### Be Specific
```
✅ Good: "Fixed IndexError in train_models_cross_multi.py line 145 when batch_size=1"
❌ Bad: "Fixed bug"
```

### Include Metrics
```
✅ Good: "Optimized data loader: 850 → 2100 samples/sec (+147%)"
❌ Bad: "Made it faster"
```

### Document Breaking Changes
Always call out compatibility issues:
```
⚠️ Breaking: Renamed `eval_legacy.py` to `eval.py`. Update import statements.
Migration: Replace `from eval_legacy import` with `from eval import`
```

### Link Related Items
```
Related:
- Issue #42: https://github.com/org/repo/issues/42
- PR #45: https://github.com/org/repo/pull/45
- Discussion: docs/design/multi_gpu.md
```

### Keep It Current
- Add entries immediately after making changes
- Don't batch entries from multiple days
- Review and update before releases

## 🔄 Workflow Example

1. **Daily Development** → Update CODELOG.md as you code
2. **Data Collection Sprint** → Update DATALOG.md with new samples
3. **Hardware Upgrade** → Document in HARDWARE.md
4. **Release Preparation** → Summarize all logs into CHANGELOG.md
5. **Version Tag** → Create git tag matching version in CHANGELOG.md

## 📝 Version Numbers

### Code (Semantic Versioning)
- **MAJOR**: Breaking API changes (1.0.0 → 2.0.0)
- **MINOR**: New features, backward compatible (1.0.0 → 1.1.0)
- **PATCH**: Bug fixes, backward compatible (1.0.0 → 1.0.1)

### Dataset (Semantic Versioning)
- **MAJOR**: Incompatible schema changes
- **MINOR**: New data/features added
- **PATCH**: Bug fixes/corrections

## ❓ FAQ

**Q: Do I need to update all files for every change?**  
A: No, only update the relevant file(s). Most changes only need one file.

**Q: Should I update CHANGELOG.md for small bug fixes?**  
A: Not immediately. Update CODELOG.md first, then summarize in CHANGELOG.md when preparing a release.

**Q: What if I'm not sure which file to use?**  
A: When in doubt, use CODELOG.md for code changes or DATALOG.md for data changes. You can always refactor later.

**Q: Can I modify the templates?**  
A: Yes! These are starting points. Adapt them to your team's needs, but keep the core sections.

**Q: How much detail should I include?**  
A: Enough for someone else (or future you) to understand what changed and why without reading the code. Include context, not just facts.

## 📚 Additional Resources

- [Keep a Changelog](https://keepachangelog.com/) - Industry standard format
- [Semantic Versioning](https://semver.org/) - Version numbering best practices
- [Conventional Commits](https://www.conventionalcommits.org/) - Git commit message format

---

**Remember**: Good logging saves time and prevents confusion. Future you (and your teammates) will thank present you! 🙏
