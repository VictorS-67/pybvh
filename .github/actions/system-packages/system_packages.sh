#!/usr/bin/env bash
# Make sure the runner has the named Ubuntu packages, and fetch from the
# package mirror only what the runner image lacks and the cache does not hold.
#
#   system_packages.sh probe PACKAGE...    write missing=, deb-dir= and
#                                          cache-key= to $GITHUB_OUTPUT
#   system_packages.sh install PACKAGE...  install the missing ones; set
#                                          CACHE_HIT=true when deb-dir holds
#                                          their .deb files from the cache
#
# The action in this folder runs both, around a restore and a save of
# deb-dir with actions/cache.
#
# Why: on a slow mirror, apt-get once spent 29 of a job's 30 minutes
# downloading ffmpeg (issue #195). A package already on the image costs no
# mirror call, a cache hit costs none either, and every mirror call has its
# own time limit, so a slow mirror fails this step with a message that says
# so instead of cancelling the job.
#
# The cache key holds the image's OS and version: the .deb files were
# resolved against that image's installed packages, and a newer image may
# hold newer ones.
set -euo pipefail

DEB_DIR="$HOME/.cache/system-package-debs"

# Each mirror call is bounded. Unbounded, a mirror that trickles at a few
# kB/s holds apt-get until the job's own limit cancels the job: the image's
# apt timeouts (Acquire::http::Timeout) catch a stalled connection, not a
# slow one. A normal run takes seconds for either call.
UPDATE_LIMIT_MINUTES=2
DOWNLOAD_LIMIT_MINUTES=4
# A second download opens new connections, which may reach a faster mirror
# host, and resumes the partial files the first one left.
DOWNLOAD_ATTEMPTS=2

# How to tell that a package is on the runner: the command or the shared
# library the jobs need from it. dpkg cannot tell by name alone: on Ubuntu
# 24.04, libglib2.0-0 is installed as libglib2.0-0t64.
is_present() {
    case "$1" in
        ffmpeg) command -v ffmpeg >/dev/null ;;
        xvfb) command -v xvfb-run >/dev/null ;;
        libgl1) has_library libGL.so.1 ;;
        libglib2.0-0) has_library libglib-2.0.so.0 ;;
        libxrender1) has_library libXrender.so.1 ;;
        libsm6) has_library libSM.so.6 ;;
        *)
            echo "::error::system_packages.sh has no probe for '$1'; add one to is_present." >&2
            exit 2
            ;;
    esac
}

has_library() {
    # Not grep -q: its early exit would end ldconfig with SIGPIPE, and
    # pipefail would report the library missing.
    /sbin/ldconfig -p | grep -F "$1 (" >/dev/null
}

missing_of() {
    local package missing=()
    for package in "$@"; do
        is_present "$package" || missing+=("$package")
    done
    echo "${missing[*]}"
}

# Run a mirror call under its time limit: bounded LIMIT_MINUTES CALL...
# Return the call's status; timeout's 124 means the limit ran out.
bounded() {
    local limit="$1"
    shift
    local status=0
    timeout "${limit}m" "$@" || status=$?
    if ((status == 124)); then
        echo "::warning::$* ran out of its ${limit} minutes."
    fi
    return "$status"
}

mirror_failure() {
    echo "::error title=Package mirror slow or unreachable::$1 The Ubuntu package mirror is slow or unreachable; no test or notebook has run yet. Rerun the job later."
    exit 1
}

# Download the .deb files of the packages and of every dependency the image
# lacks into DEB_DIR, from the mirror.
download() {
    sudo rm -rf "$DEB_DIR"
    mkdir -p "$DEB_DIR/partial"
    bounded "$UPDATE_LIMIT_MINUTES" sudo apt-get update \
        || mirror_failure "apt-get update failed or did not finish within $UPDATE_LIMIT_MINUTES minutes."
    local attempt
    for ((attempt = 1; attempt <= DOWNLOAD_ATTEMPTS; attempt++)); do
        # The archive folder sits in the runner's home, which apt's _apt
        # user cannot enter: apt downloads as root.
        if bounded "$DOWNLOAD_LIMIT_MINUTES" sudo apt-get install --download-only -y \
            --no-install-recommends \
            -o Dir::Cache::Archives="$DEB_DIR" \
            -o APT::Sandbox::User=root \
            "$@"; then
            # actions/cache archives the folder as the runner user.
            sudo rm -rf "$DEB_DIR/partial" "$DEB_DIR/lock"
            sudo chown -R "$(id -u):$(id -g)" "$DEB_DIR"
            return 0
        fi
        echo "::warning::Download attempt $attempt of $DOWNLOAD_ATTEMPTS from the package mirror failed."
    done
    mirror_failure "apt-get did not download $* in $DOWNLOAD_ATTEMPTS attempts of at most $DOWNLOAD_LIMIT_MINUTES minutes each."
}

# Install the .deb files in DEB_DIR and nothing else. With an empty package
# list, apt resolves dependencies against the installed packages and these
# files alone, so it cannot reach for the mirror.
install_downloaded() {
    local no_lists
    no_lists="$(mktemp -d)"
    mkdir -p "$no_lists/partial"
    sudo apt-get install -y --no-install-recommends \
        -o Dir::State::Lists="$no_lists" \
        "$DEB_DIR"/*.deb
}

probe() {
    local missing
    missing="$(missing_of "$@")"
    local key
    key="system-packages-${ImageOS:-unknown}-${ImageVersion:-unknown}-$(uname -m)-${missing// /-}"
    echo "Missing from the runner image: ${missing:-nothing}"
    {
        echo "missing=$missing"
        echo "deb-dir=$DEB_DIR"
        echo "cache-key=$key"
    } >>"$GITHUB_OUTPUT"
}

install() {
    if [[ "${CACHE_HIT:-}" == "true" ]]; then
        echo "Installing from the cache, without the package mirror."
        if ! install_downloaded; then
            echo "::warning::The cached packages did not install; downloading them from the mirror."
            download "$@"
            install_downloaded
        fi
    else
        download "$@"
        install_downloaded
    fi
    local still_missing
    still_missing="$(missing_of "$@")"
    if [[ -n "$still_missing" ]]; then
        echo "::error::apt-get finished, but these packages are still missing: $still_missing"
        exit 1
    fi
}

command="${1:?usage: system_packages.sh probe|install PACKAGE...}"
shift
case "$command" in
    probe) probe "$@" ;;
    install) install "$@" ;;
    *)
        echo "usage: system_packages.sh probe|install PACKAGE..." >&2
        exit 2
        ;;
esac
