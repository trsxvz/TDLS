#!/bin/sh
# Builds the documentation site for every version: origin/main and each
# release tag, each into its own directory of the output, with its
# Doxygen API reference under api/. The root of the output redirects to
# main. Every version is built from its own sources, so a release keeps
# the documentation it shipped with.
#
#   docs/build_site.sh <output directory>
#
# DOXYGEN names the doxygen binary (default: doxygen on the PATH).
set -eu
out=$(realpath -m "$1")
doxygen=${DOXYGEN:-doxygen}
repo=$(git rev-parse --show-toplevel)
versions="main $(git tag --list 'v*' --sort=-version:refname)"

rm -rf "$out"
mkdir -p "$out"
for v in $versions; do
    if [ "$v" = main ]; then ref=origin/main; else ref=$v; fi
    tree=$(mktemp -d)
    git -C "$repo" worktree add --detach "$tree" "$ref" > /dev/null
    # doxygen warnings are the business of the doxygen workflow, not of the site
    (cd "$tree/docs" && mkdir -p _build && "$doxygen" Doxyfile > /dev/null 2>&1)
    # warnings block main, not a frozen release built by a newer Sphinx
    if [ "$v" = main ]; then strict=-W; else strict=; fi
    TDLS_DOCS_VERSION=$v TDLS_DOCS_VERSIONS="$versions" \
        sphinx-build $strict -q -b html "$tree/docs/sphinx" "$out/$v"
    cp -r "$tree/docs/_build/doxygen/html" "$out/$v/api"
    git -C "$repo" worktree remove --force "$tree"
done
cat > "$out/index.html" <<'HTML'
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="0; url=main/">
  <link rel="canonical" href="main/">
  <title>TDLS documentation</title>
</head>
<body><a href="main/">TDLS documentation</a></body>
</html>
HTML
echo "site built for: $versions"
