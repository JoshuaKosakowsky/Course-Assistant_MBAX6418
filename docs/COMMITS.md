# Commit Message Conventions

Follow this for every commit in this repo. Enforced by review (and by
habit); keep the discipline even on small commits.

## Commit types

| Type       | Meaning                                                                 |
|------------|-------------------------------------------------------------------------|
| `feat:`    | Adds a new feature                                                      |
| `fix:`     | Fixes a bug                                                             |
| `refactor:`| Rewrites/restructures code; neither fixes a bug nor adds a feature      |
| `chore:`   | Not a fix or feature; doesn't modify src or test (e.g. deps, .gitignore)|
| `perf:`    | Special refactor aimed at improving performance                         |
| `ci:`      | Continuous integration related                                          |
| `ops:`     | Operational components: infrastructure, deployment, backup, recovery    |
| `build:`   | Build system, build tool, CI pipeline, dependencies, project version    |
| `docs:`    | Documentation (README, docs/...)                                        |
| `style:`   | Formatting only: whitespace, missing semicolons — no behavior change    |
| `revert:`  | Reverts a previous commit                                               |
| `test:`    | Adds missing tests or corrects existing tests                           |

## Rules for a great commit message

1. Limit the subject line to **50 characters**
2. **Capitalize** the subject line
3. Do **not** end the subject line with a period
4. Separate the subject from the body with a **blank line**
5. Wrap the body at **72 characters**
6. Use the body to explain **what and why** (not how — the diff shows how)
7. Use the **imperative mood** — the subject reads like a command
   ("feat: Add unit tests for user authentication")
8. One logical change per commit

## Examples

Good:

```text
feat: Add topic filter to quiz generation

Allow students to scope quiz questions to a chosen lecture or
module. Candidate questions are filtered by the chunk-level
topic metadata recorded at ingest time.
```

```text
fix: Preserve source metadata across chunk boundaries

Text splitters broke page attribution when a heading landed at
the end of a chunk; keep the page/slide field on every chunk.
```

Bad:

```text
fixed stuff
```

```text
Update files and stuff and other things also this subject is way
too long and keeps going
```

## Branches and PRs

- One issue = one branch = one PR.
- Branch names mirror commit types: `feat/<slug>`, `fix/<slug>`,
  `test/<slug>`, `docs/<slug>`.
- PR titles follow the same rules as commit subjects.
- Link the issue in the PR body: `Closes #N` closes it on merge.
