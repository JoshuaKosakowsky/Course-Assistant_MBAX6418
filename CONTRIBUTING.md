# Contributing — Course Assistant (MBAX 6418-801)

Thanks for contributing. This is a team repo with a grader involved: the
assignment requires GitHub issues, branches, pull requests, reviews, and
repeatable tests. AI agents should follow AGENTS.md; this file is the
workflow for everyone.

## Ground rules

- **Security:** real API keys live only in the local `.env` (gitignored).
  Never put them in code, logs, errors, screenshots, issues, PRs, docs, or
  chat. `.env.example` holds dummy placeholders and is the only tracked
  config template. CI and tests enforce this.
- **Honesty:** the application never invents answers or citations. It
  answers from retrieved evidence and says so when material is missing.

## Workflow

1. **Claim an issue.** Substantial tasks are filed as issues with completion
   checks (currently #1–#9). Comment on the issue to take ownership; the
   team decides who does what.
2. **Branch.** Create a branch from `main`, named `type/slug` matching the
   issue (e.g. `feat/ingest`). One issue = one branch = one PR — this keeps
   review small and each bug fixed independently.
3. **Commit.** Follow docs/COMMITS.md exactly:

   - Subject ≤ 50 characters, capitalized, no trailing period, imperative
     mood (`feat: Add topic filter to quiz generation`)
   - Blank line, then a body explaining what and why, wrapped at 72 chars
   - One logical change per commit

4. **Test.** Add/adjust tests with your change, per docs/TESTING.md. At
   minimum, `pytest` (unit + smoke) must pass before opening a PR.
5. **Push and open a PR.** Use the PR template; link the issue (`Closes #N`
   in the body closes it automatically on merge).
6. **Review.** At least one teammate reviews and approves before merge.
   Address review comments in follow-up commits on the same branch.

## Reporting bugs

- **One bug = one issue.** Open a separate bug report for each problem.
- Use the bug report template (.github/ISSUE_TEMPLATE/bug_report.md): a
  unique ~10-word summary (problem, not proposed fix), whether you can
  reproduce at will / occasionally / not at all, precise steps, expected vs
  actual results, and environment. Separate facts from speculation.
- Check the existing issues first; if the bug is already reported, add
  information to that issue instead of filing a duplicate.
- If you can't reproduce it but have unique details, include them anyway —
  they can be exactly what a developer needs.

## Reviewing

- Check: does the PR match its issue? Do tests pass? Any secrets in the
  diff? Docs updated if behavior changed? Is the honest "tested vs
  unchecked" statement present?
- Be specific and kind; suggest, don't demand.

## Definition of done

The PR template checklist mirrors this: issue linked, tests green, no
secrets, screenshots for UI changes, docs updated, what-was-tested stated.
