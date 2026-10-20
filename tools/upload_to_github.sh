#!/usr/bin/env bash
#
# Test, commit and upload SuperConga to the fogelstrom-lab GitHub organisation.
#
# This mirrors FermiForge's tools/upload_to_git.sh. The original SuperConga
# remote on GitLab is left untouched as "origin"; GitHub is a second remote
# (default name "github") so that upstream history can still be pulled.
#
# Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
# See LICENSE.txt for license information.

set -euo pipefail

readonly script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly project_root="$(cd "${script_directory}/.." && pwd)"
readonly default_branch="${SUPERCONGA_BRANCH:-main}"
readonly remote_name="${SUPERCONGA_REMOTE_NAME:-github}"
readonly default_remote_url="git@github.com:fogelstrom-lab/superconga.git"
readonly maximum_file_bytes="${SUPERCONGA_MAX_FILE_BYTES:-26214400}"

branch="${default_branch}"
remote_url="${SUPERCONGA_REMOTE_URL:-${default_remote_url}}"
commit_message=""
run_tests=true
push_changes=true
dry_run=false
replace_remote=false
assume_yes=false
new_commit_created=false

report_failed_upload() {
  local status=$?
  if ((status == 0)); then
    return
  fi
  printf '\nUpload script stopped with status %s.\n' "${status}" >&2
  if "${new_commit_created}"; then
    printf 'A local commit was created, but the remote push did not complete: %s\n' \
      "$(git rev-parse --short HEAD 2>/dev/null || printf unknown)" >&2
  else
    printf 'No new commit or push was completed; any staged changes remain intact.\n' >&2
  fi
  trap - EXIT
  exit "${status}"
}
trap report_failed_upload EXIT

usage() {
  cat <<'EOF'
Usage: tools/upload_to_github.sh [options]

Build the CPU backend, run the test suite, commit, and upload SuperConga.

Default destination:
  git@github.com:fogelstrom-lab/superconga.git  (remote "github", branch main)

Options:
  -m, --message TEXT       Commit message. A dated message is used by default.
      --remote-url URL     Override the Git remote URL.
      --github             Use git@github.com:fogelstrom-lab/superconga.git.
      --https              Use the GitHub HTTPS URL instead of SSH.
      --replace-remote     Replace the URL of an existing "github" remote.
      --branch NAME        Branch to upload (default: main).
      --skip-tests         Do not build and run the CPU test suite first.
      --no-push            Commit locally but do not contact or push the remote.
      --dry-run            Run checks and preview files; do not stage or commit.
  -y, --yes                Do not ask for final interactive confirmation.
  -h, --help               Show this help.

Environment overrides:
  SUPERCONGA_REMOTE_URL, SUPERCONGA_BRANCH, SUPERCONGA_REMOTE_NAME,
  SUPERCONGA_BUILD_JOBS, SUPERCONGA_MAX_FILE_BYTES, OMP_NUM_THREADS.
EOF
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

while (($# > 0)); do
  case "$1" in
    -m|--message) (($# >= 2)) || fail "$1 requires an argument"; commit_message="$2"; shift 2 ;;
    --remote-url) (($# >= 2)) || fail "$1 requires an argument"; remote_url="$2"; shift 2 ;;
    --github) remote_url="${default_remote_url}"; shift ;;
    --https) remote_url="https://github.com/fogelstrom-lab/superconga.git"; shift ;;
    --replace-remote) replace_remote=true; shift ;;
    --branch) (($# >= 2)) || fail "$1 requires an argument"; branch="$2"; shift 2 ;;
    --skip-tests) run_tests=false; shift ;;
    --no-push) push_changes=false; shift ;;
    --dry-run) dry_run=true; push_changes=false; shift ;;
    -y|--yes) assume_yes=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown option: $1" ;;
  esac
done

cd "${project_root}"
[[ -f CMakeLists.txt && -f superconga.py && -d include/conga && -d tests ]] || \
  fail "${project_root} does not look like the SuperConga source root"
[[ -d .git ]] || fail "no git repository here"

current_branch="$(git symbolic-ref --quiet --short HEAD 2>/dev/null || true)"
[[ -n "${current_branch}" ]] || fail "detached HEAD is not supported by the upload script"
if [[ "${current_branch}" != "${branch}" ]]; then
  fail "current branch is ${current_branch}; switch to ${branch} (git checkout -b ${branch}) or pass --branch ${current_branch}"
fi

if "${run_tests}"; then
  cmake_command="$(command -v cmake || true)"
  [[ -n "${cmake_command}" ]] || cmake_command=/opt/homebrew/bin/cmake
  [[ -x "${cmake_command}" ]] || fail "CMake not found"
  ctest_command="$(dirname "${cmake_command}")/ctest"
  build_directory="${project_root}/build/git-upload-check"
  printf 'Building the CPU backend and running the test suite in %s\n' "${build_directory}"
  "${cmake_command}" -S "${project_root}" -B "${build_directory}" -G Ninja \
    -DCONGA_BACKEND=CPU -DCMAKE_BUILD_TYPE=Release \
    -DCONGA_BUILD_TESTS=ON -DCONGA_VISUALIZE=OFF
  "${cmake_command}" --build "${build_directory}" \
    --parallel "${SUPERCONGA_BUILD_JOBS:-8}"
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
    "${ctest_command}" --test-dir "${build_directory}" --output-on-failure
  python3 -m py_compile superconga.py frontend/*/*.py tools/*.py
  printf 'Verification passed.\n'
fi

printf '\nGit working-tree preview:\n'
git status --short

if "${dry_run}"; then
  printf '\nFiles Git would add or update:\n'
  git add --dry-run -A
  printf '\nDry run complete; nothing was staged, committed, or pushed.\n'
  exit 0
fi

git add -A

staged_file_is_too_large=false
while IFS= read -r -d '' path; do
  [[ -f "${path}" ]] || continue
  if stat -f '%z' "${path}" >/dev/null 2>&1; then
    file_bytes="$(stat -f '%z' "${path}")"
  else
    file_bytes="$(stat -c '%s' "${path}")"
  fi
  if ((file_bytes > maximum_file_bytes)); then
    printf 'error: staged file exceeds %s bytes: %s (%s bytes)\n' \
      "${maximum_file_bytes}" "${path}" "${file_bytes}" >&2
    staged_file_is_too_large=true
  fi
done < <(git diff --cached --name-only --diff-filter=ACMR -z)
"${staged_file_is_too_large}" && \
  fail "move run data to private/ or external storage (see .gitignore)"

printf '\nStaged change summary:\n'
git diff --cached --stat

if "${push_changes}"; then
  printf '\nPlanned destination: %s (%s branch, remote %s)\n' \
    "${remote_url}" "${branch}" "${remote_name}"
else
  printf '\nNo remote push will be made.\n'
fi

if ! "${assume_yes}"; then
  [[ -t 0 ]] || fail "interactive confirmation is unavailable; rerun with --yes"
  printf 'Continue with the commit and%s? [y/N] ' \
    "$(if "${push_changes}"; then printf ' push'; fi)"
  read -r reply
  case "${reply}" in
    y|Y|yes|YES|Yes) ;;
    *) printf 'Stopped before committing. Staged files remain available for inspection.\n'; exit 0 ;;
  esac
fi

if ! git diff --cached --quiet; then
  [[ -n "$(git config --get user.name || true)" ]] || fail "Git user.name is not configured"
  [[ -n "$(git config --get user.email || true)" ]] || fail "Git user.email is not configured"
  if [[ -z "${commit_message}" ]]; then
    commit_message="SuperConga update $(date '+%Y-%m-%d %H:%M %Z')"
  fi
  git commit -m "${commit_message}"
  new_commit_created=true
  printf 'Created local commit %s\n' "$(git rev-parse --short HEAD)"
else
  printf 'No staged changes; no new commit was created.\n'
fi

if ! "${push_changes}"; then
  printf 'Local commit complete; remote upload was skipped.\n'
  exit 0
fi

if git remote get-url "${remote_name}" >/dev/null 2>&1; then
  configured_url="$(git remote get-url "${remote_name}")"
  if [[ "${configured_url}" != "${remote_url}" ]]; then
    "${replace_remote}" || fail "${remote_name} is ${configured_url}, not ${remote_url}; use --replace-remote to change it"
    git remote set-url "${remote_name}" "${remote_url}"
  fi
else
  git remote add "${remote_name}" "${remote_url}"
fi

printf 'Checking remote %s (%s)\n' "${remote_name}" "${remote_url}"
git ls-remote "${remote_name}" >/dev/null
if git ls-remote --exit-code --heads "${remote_name}" "refs/heads/${branch}" >/dev/null 2>&1; then
  git fetch --no-tags "${remote_name}" "${branch}"
  if ! git merge-base --is-ancestor "${remote_name}/${branch}" HEAD; then
    fail "remote ${branch} is not an ancestor of local HEAD; reconcile it explicitly (the script never force-pushes)"
  fi
fi

git push --set-upstream "${remote_name}" "${branch}"
printf 'Uploaded %s at commit %s to %s without rewriting remote history.\n' \
  "${branch}" "$(git rev-parse --short HEAD)" "${remote_url}"
