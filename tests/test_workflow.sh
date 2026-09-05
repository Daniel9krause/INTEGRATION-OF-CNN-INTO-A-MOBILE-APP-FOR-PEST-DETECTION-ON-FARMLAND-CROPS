#!/usr/bin/env bash
# test_workflow
# --------------
# Exercises the shell logic inside .github/workflows/build.yml locally, with
# stubbed `curl` and `buildozer`.
#
# This exists because the APK build is a 20-minute round trip through GitHub
# Actions, and a mistake in a `run:` block is only discovered at the end of
# it. Two real outages of download.savannah.gnu.org - which p4a's freetype
# recipe hardcodes - cost two red builds on an unmodified tree, and the fix
# for them is itself shell script that can be wrong.
#
# Both scripts under test are EXTRACTED FROM build.yml rather than copied
# here, so the workflow and this test cannot drift apart: change the
# workflow, and this tests the change.
#
# The digest check is exercised for real - the stubbed curl serves the
# genuine freetype tarball and the step's own sha256sum runs untouched. The
# tarball is cached under tests/_cache/ (gitignored) and downloaded on first
# run; if it cannot be fetched, the prefetch section skips and says so.
#
# The prefetch assertions go through p4a_verdict(), a transcription of
# python-for-android's own Recipe.download() acceptance rule, rather than
# just checking that a file landed somewhere. The first version of this test
# asserted "the tarball is in the right directory", which passed while the
# real build still re-downloaded: p4a treats a tarball with no `.mark-` file
# beside it as an interrupted download and DELETES it. Asserting that our
# own script did what we told it to proves nothing about whether the tool
# downstream will accept the result - so the rule that actually decides it
# is modelled here, and pinned to the source it came from.
#
# Run:  bash tests/test_workflow.sh

set -u

TESTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$TESTS_DIR")"
WORKFLOW="$REPO/.github/workflows/build.yml"
CACHE="$TESTS_DIR/_cache"
WORK="$CACHE/wf_test"

FT_VERSION=2.14.1
FT_SHA256=174d9e53402e1bf9ec7277e22ec199ba3e55a6be2c0740cb18c0ee9850fc8c34
REAL_TARBALL="$CACHE/freetype-$FT_VERSION.tar.gz"

pass=0; fail=0; skip=0
check() { if [ "$2" = "$3" ]; then echo "  PASS  $1"; pass=$((pass+1));
          else echo "  FAIL  $1 (got '$2', want '$3')"; fail=$((fail+1)); fi; }

echo "============================================================"
echo "build workflow tests"
echo "============================================================"

mkdir -p "$CACHE"
rm -rf "$WORK"; mkdir -p "$WORK/bin"

# --- pull the two run: blocks straight out of the workflow ---------------
if ! python - "$WORKFLOW" "$WORK" <<'PY'
import io, sys
try:
    import yaml
except ImportError:
    sys.exit("PyYAML not installed - pip install pyyaml")
wf = yaml.safe_load(io.open(sys.argv[1], encoding="utf-8"))
steps = {s["name"]: s for s in wf["jobs"]["build"]["steps"]}
for name, fn in (("Prefetch freetype from a working mirror", "prefetch.sh"),
                 ("Build APK with Buildozer", "build.sh")):
    if name not in steps:
        sys.exit(f"step not found in build.yml: {name!r}")
    io.open(f"{sys.argv[2]}/{fn}", "w", encoding="utf-8",
            newline="\n").write(steps[name]["run"])
PY
then
    echo "  FAIL  could not extract the run: blocks from build.yml"
    exit 1
fi
echo "  PASS  build.yml parses and both run: blocks extract"
pass=$((pass+1))

for f in prefetch.sh build.sh; do
    if bash -n "$WORK/$f"; then
        echo "  PASS  $f is valid bash"; pass=$((pass+1))
    else
        echo "  FAIL  $f is not valid bash"; fail=$((fail+1))
    fi
done

# The retry loop's real backoff is 60s then 120s. Neutralise the sleep only;
# the arithmetic that computes it stays, and is asserted below.
sed -i 's/^\( *\)sleep "\$backoff"/\1echo "TEST-BACKOFF=$backoff"/' "$WORK/build.sh"

# GitHub runs `run:` blocks as `bash -e`, with no pipefail. Match it exactly:
# the difference is the whole point of one of the assertions below.
run() { ( cd "$WORK/repo" && PATH="$WORK/bin:$PATH" bash -e "$WORK/$1" ); }

fresh_repo() {
    rm -rf "$WORK/repo"; mkdir -p "$WORK/repo"
    printf 'android.api = 33\nandroid.archs = arm64-v8a\n' > "$WORK/repo/buildozer.spec"
}

seeded_path() {
    echo "$WORK/repo/.buildozer/android/platform/build-$1/packages/freetype/freetype-$FT_VERSION.tar.gz"
}

# A faithful transcription of python-for-android's Recipe.download(), from
# pythonforandroid/recipe.py at the p4a version buildozer.spec pins
# (p4a.commit = v2026.05.09; identical on develop):
#
#     do_download = True
#     marker_filename = '.mark-{}'.format(filename)
#     if exists(filename) and isfile(filename):
#         if not exists(marker_filename):
#             shprint(sh.rm, filename)
#         else:
#             ...verify digests (the freetype recipe declares none)...
#             do_download = False
#
# Echoes "download" or "use-cached", and performs the same rm side effect,
# so a test can assert both the decision and that the file survives it.
p4a_verdict() {
    local tarball="$1"
    local marker
    marker="$(dirname "$tarball")/.mark-$(basename "$tarball")"
    if [ -f "$tarball" ]; then
        if [ ! -f "$marker" ]; then
            rm -f "$tarball"          # p4a reads this as a partial download
            echo "download"
        else
            echo "use-cached"
        fi
    else
        echo "download"
    fi
}

# --- the prefetch step ---------------------------------------------------
echo
echo "prefetch: mirror fallback and the digest check p4a does not do"

if [ ! -f "$REAL_TARBALL" ]; then
    echo "  ....  fetching the real tarball once into tests/_cache/"
    for url in \
        "https://downloads.sourceforge.net/project/freetype/freetype2/$FT_VERSION/freetype-$FT_VERSION.tar.gz" \
        "https://mirror.csclub.uwaterloo.ca/nongnu/freetype/freetype-$FT_VERSION.tar.gz"; do
        curl -fsSL --retry 3 --retry-all-errors --max-time 300 -o "$REAL_TARBALL" "$url" && break
        rm -f "$REAL_TARBALL"
    done
fi

if [ ! -f "$REAL_TARBALL" ] || \
   ! echo "$FT_SHA256  $REAL_TARBALL" | sha256sum -c - > /dev/null 2>&1; then
    rm -f "$REAL_TARBALL"
    echo "  SKIP  prefetch tests - could not fetch a verified freetype tarball"
    echo "        (offline, or every mirror is down; the retry tests still run)"
    skip=$((skip+1))
else
    # A curl stub that serves whichever mirror the scenario nominates.
    cat > "$WORK/bin/curl" <<'EOF'
#!/usr/bin/env bash
out=""; url=""
while [ $# -gt 0 ]; do
  case "$1" in
    -o) out="$2"; shift 2 ;;
    http*) url="$1"; shift ;;
    *) shift ;;
  esac
done
case "$url" in
  *"$GOOD_MIRROR"*) cp "$REAL_TARBALL" "$out"; exit 0 ;;
  *"${CORRUPT_MIRROR:-__none__}"*) echo "not a tarball" > "$out"; exit 0 ;;
  *) exit 22 ;;   # curl's exit code for an HTTP error under -f
esac
EOF
    chmod +x "$WORK/bin/curl"
    export REAL_TARBALL

    # 1. The happy path.
    fresh_repo
    export GOOD_MIRROR="sourceforge.net" CORRUPT_MIRROR=""
    out=$(run prefetch.sh 2>&1); check "exits 0 when the first mirror works" "$?" "0"
    s=$(seeded_path arm64-v8a)
    check "seeds the tarball where p4a looks for it" "$([ -f "$s" ] && echo yes || echo no)" "yes"
    check "the seeded file is byte-identical to the real one" \
          "$(cmp -s "$s" "$REAL_TARBALL" && echo same || echo differs)" "same"
    # The assertion that matters, and the one the first version of this
    # test was missing: p4a must ACCEPT what we seeded.
    check "seeds the .mark- marker p4a requires" \
          "$([ -f "$(dirname "$s")/.mark-$(basename "$s")" ] && echo yes || echo no)" "yes"
    check "p4a would skip the download entirely" "$(p4a_verdict "$s")" "use-cached"
    check "and the seeded file survives p4a's partial-download check" \
          "$([ -f "$s" ] && echo yes || echo no)" "yes"

    # 2. The outage that started all this: first mirror dead, fall through.
    fresh_repo
    export GOOD_MIRROR="uwaterloo.ca" CORRUPT_MIRROR=""
    out=$(run prefetch.sh 2>&1); check "exits 0 when it has to fall through" "$?" "0"
    check "still seeds the tarball" "$([ -f "$(seeded_path arm64-v8a)" ] && echo yes || echo no)" "yes"
    check "p4a accepts the fallback mirror's copy too" \
          "$(p4a_verdict "$(seeded_path arm64-v8a)")" "use-cached"
    check "logs the dead mirror" "$(echo "$out" | grep -c 'unusable')" "1"

    # 3. The security-relevant one. p4a's freetype recipe declares no
    #    checksum, so nothing downstream would notice a substituted file.
    fresh_repo
    export GOOD_MIRROR="uwaterloo.ca" CORRUPT_MIRROR="sourceforge.net"
    out=$(run prefetch.sh 2>&1); check "exits 0 after rejecting a bad file" "$?" "0"
    check "a digest mismatch is never seeded" \
          "$(cmp -s "$(seeded_path arm64-v8a)" "$REAL_TARBALL" && echo same || echo differs)" "same"
    check "logs the rejection" "$(echo "$out" | grep -c 'unusable')" "1"

    # 4. Every mirror down. This step is a safety net, so it must not itself
    #    become a new way for the build to fail.
    fresh_repo
    export GOOD_MIRROR="__nothing__" CORRUPT_MIRROR=""
    out=$(run prefetch.sh 2>&1); check "a total outage does not fail the build" "$?" "0"
    check "seeds nothing" "$([ -f "$(seeded_path arm64-v8a)" ] && echo yes || echo no)" "no"
    check "warns, so the log says why" "$(echo "$out" | grep -c '::warning::')" "1"

    # 5. The arch list comes from the spec, not a hardcoded string.
    fresh_repo
    printf 'android.archs = armeabi-v7a, arm64-v8a\n' > "$WORK/repo/buildozer.spec"
    export GOOD_MIRROR="sourceforge.net" CORRUPT_MIRROR=""
    run prefetch.sh > /dev/null 2>&1
    for a in armeabi-v7a arm64-v8a; do
        check "seeds $a, as named in buildozer.spec" \
              "$([ -f "$(seeded_path "$a")" ] && echo yes || echo no)" "yes"
        check "p4a would use the cached copy for $a" \
              "$(p4a_verdict "$(seeded_path "$a")")" "use-cached"
    done
fi

    # 6. The regression, stated directly. A tarball seeded WITHOUT its
    #    marker is deleted by p4a and re-downloaded from the dead mirror -
    #    which is precisely how the first attempt at this step failed, and
    #    it failed invisibly, because the file really was in the right
    #    place. If someone drops the `touch` from the workflow, this is the
    #    assertion that catches it.
    fresh_repo
    export GOOD_MIRROR="sourceforge.net" CORRUPT_MIRROR=""
    run prefetch.sh > /dev/null 2>&1
    s=$(seeded_path arm64-v8a)
    rm -f "$(dirname "$s")/.mark-$(basename "$s")"
    check "without the marker, p4a re-downloads" "$(p4a_verdict "$s")" "download"
    check "and deletes the seeded file on the way" \
          "$([ -f "$s" ] && echo yes || echo no)" "no"

# --- the retry loop ------------------------------------------------------
echo
echo "build: retry a flaky mirror, never a broken build"

stub() { cat > "$WORK/bin/buildozer"; chmod +x "$WORK/bin/buildozer"; }
fresh_repo

# 6. A clean build. The stub exits without draining stdin, so `yes` dies of
#    SIGPIPE - which is exactly why that step must NOT use `set -o pipefail`.
#    With pipefail this assertion fails and every green build reads as red.
stub <<'EOF'
#!/usr/bin/env bash
echo "BUILD SUCCESSFUL"
exit 0
EOF
out=$(run build.sh 2>&1); check "a clean build exits 0 despite yes/SIGPIPE" "$?" "0"
check "names the winning attempt" "$(echo "$out" | grep -c 'succeeded on attempt 1')" "1"

# 7. A mirror blip that clears on its own.
echo 0 > "$WORK/bin/n"
stub <<'EOF'
#!/usr/bin/env bash
d=$(dirname "$0"); n=$(cat "$d/n"); n=$((n+1)); echo "$n" > "$d/n"
if [ "$n" -lt 2 ]; then
  echo "urllib.error.HTTPError: HTTP Error 502: Bad Gateway"; exit 1
fi
echo "BUILD SUCCESSFUL"; exit 0
EOF
out=$(run build.sh 2>&1); check "recovers from a transient download failure" "$?" "0"
check "retried exactly once" "$(echo "$out" | grep -c 'died fetching a dependency')" "1"
check "first backoff is 60s" "$(echo "$out" | grep -c 'TEST-BACKOFF=60')" "1"

# 8. What the grep guard is for: retrying a genuine build error three times
#    turns a 4-minute red build into a 40-minute one and buries the message.
stub <<'EOF'
#!/usr/bin/env bash
echo "ninja: build stopped: subcommand failed."; exit 1
EOF
out=$(run build.sh 2>&1); check "a genuine build error fails" "$?" "1"
check "and is not retried" "$(echo "$out" | grep -c 'died fetching')" "0"
check "and says why it stopped" "$(echo "$out" | grep -c 'not a network fault')" "1"

# 9. A mirror that never comes back must give up rather than spin.
stub <<'EOF'
#!/usr/bin/env bash
echo "Download failed: HTTP Error 504: Gateway Time-out"; exit 1
EOF
out=$(run build.sh 2>&1); check "gives up on a permanent outage" "$?" "1"
check "after 3 attempts" "$(echo "$out" | grep -c 'attempt 3 of 3')" "1"
check "with an escalating backoff" "$(echo "$out" | grep -c 'TEST-BACKOFF=120')" "1"

rm -rf "$WORK"

echo
echo "============================================================"
if [ "$skip" -gt 0 ]; then
    echo "$pass passed, $fail failed, $skip section(s) skipped"
else
    echo "$pass passed, $fail failed"
fi
echo "============================================================"
[ "$fail" -eq 0 ]
