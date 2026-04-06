#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
VERSION="${1:?usage: scripts/rust/release/tag-package-release.sh <version> [<ref>]}"
REF="${2:-HEAD}"
PACKAGE_FILE="Package.swift"
EXPECTED_ARTIFACT_PATH="releases/download/tokenizers-rust-${VERSION}/TokenizersRust-${VERSION}.xcframework.zip"

if [[ ! "${VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]]; then
  echo "Version must be semantic, for example 0.3.0 or 0.3.0-rc.1." >&2
  exit 1
fi

cd "${REPO_ROOT}"

git rev-parse --verify "${REF}" >/dev/null
git rev-parse --verify "refs/tags/${VERSION}" >/dev/null 2>&1 && {
  echo "Tag ${VERSION} already exists." >&2
  exit 1
}

git show "${REF}:${PACKAGE_FILE}" | grep -F "${EXPECTED_ARTIFACT_PATH}" >/dev/null || {
  echo "${PACKAGE_FILE} at ${REF} does not point at tokenizers-rust-${VERSION}." >&2
  exit 1
}

gh release view "tokenizers-rust-${VERSION}" >/dev/null 2>&1 || {
  echo "Rust artifact release tokenizers-rust-${VERSION} does not exist." >&2
  exit 1
}

git tag -a "${VERSION}" "${REF}" -m "${VERSION}"

echo
echo "Created local tag ${VERSION} at ${REF}."
echo "Push it with:"
echo "  git push origin ${VERSION}"
