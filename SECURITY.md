## Security

This ROCm-only port evaluates user-supplied GPU kernel source in subprocesses,
so security reports should focus on repository code, evaluation isolation,
schema validation, build staging, and documentation that could cause unsafe
usage.

Do not report private security issues through public GitHub issues or pull
requests. Use the repository owner's private disclosure channel when available,
or contact the maintainer directly with enough detail to reproduce the issue.

Please include:

- affected commit or version,
- affected command or workflow,
- steps to reproduce,
- proof-of-concept source or input files when safe to share,
- expected and actual impact.

Do not include credentials, proprietary kernels, Hugging Face tokens, or
downloaded benchmark datasets in a public report.
