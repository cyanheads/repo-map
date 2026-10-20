# Security Policy

## Supported Versions

Security fixes land on the latest release of `repo-map`. Older versions are not patched; upgrade to the current release.

## Reporting a Vulnerability

Please do not open a public issue for security reports. Instead:

- Report privately via GitHub: **Security** tab → **Report a vulnerability**, or
- Email **security@caseyjhand.com**

In scope: anything that sends a file to OpenRouter that the [README's upload rules](../README.md#what-gets-sent) say stays local, leaks or redirects the OpenRouter API key, or lets an analyzed repository's contents change what repo-map does beyond the text of its own report.

Include a minimal public reproduction where possible, with API keys and any private source redacted. A placeholder is enough to show the shape. You'll receive an acknowledgment, and credit in the release notes if the report leads to a fix (unless you prefer otherwise).
