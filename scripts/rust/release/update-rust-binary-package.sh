#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
VERSION="${1:?usage: scripts/rust/release/update-rust-binary-package.sh <version>}"
PACKAGE_FILE="${REPO_ROOT}/Package.swift"
TAG="tokenizers-rust-${VERSION}"
ASSET_NAME="TokenizersRust-${VERSION}.xcframework.zip"

if [[ ! "${VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]]; then
  echo "Version must be semantic, for example 0.3.0 or 0.3.0-rc.1." >&2
  exit 1
fi

release_json="$(gh release view "${TAG}" --json assets)"

mapfile -t asset_info < <(printf '%s' "${release_json}" | python3 -c '
import json
import sys

assets = json.load(sys.stdin)["assets"]
target = sys.argv[1]

for asset in assets:
    if asset["name"] == target:
        print(asset["url"])
        digest = asset.get("digest", "")
        if digest.startswith("sha256:"):
            print(digest.split(":", 1)[1])
        else:
            print("")
        raise SystemExit(0)

raise SystemExit(f"missing asset: {target}")
' "${ASSET_NAME}")

asset_url="${asset_info[0]}"
checksum="${asset_info[1]}"

if [[ -z "${checksum}" ]]; then
  temp_dir="$(mktemp -d)"
  trap 'rm -rf "${temp_dir}"' EXIT
  archive_path="${temp_dir}/${ASSET_NAME}"
  gh release download "${TAG}" --pattern "${ASSET_NAME}" --output "${archive_path}"
  checksum="$(swift package compute-checksum "${archive_path}")"
fi

python3 - <<'PY' "${PACKAGE_FILE}" "${asset_url}" "${checksum}"
import pathlib
import re
import sys

package_file = pathlib.Path(sys.argv[1])
asset_url = sys.argv[2]
checksum = sys.argv[3]
content = package_file.read_text()

content, url_count = re.subn(
    r'url: "https://github\.com/DePasqualeOrg/swift-tokenizers/releases/download/[^"]+"',
    f'url: "{asset_url}"',
    content,
    count=1,
)
content, checksum_count = re.subn(
    r'checksum: "[0-9a-f]{64}"',
    f'checksum: "{checksum}"',
    content,
    count=1,
)

if url_count != 1 or checksum_count != 1:
    raise SystemExit("failed to update Package.swift")

package_file.write_text(content)
PY

echo "Updated ${PACKAGE_FILE} to ${TAG}."
