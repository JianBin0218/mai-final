# Instructions: update the project site

You maintain `docs/index.html`, a single-file overview of this project for the team. The workflow tells you
the commit at which the page was last updated (`BASE`). Bring the page up to date with everything that
changed in `BASE..HEAD`. Edit only `docs/index.html`. Do not commit, push, or run project code.

## What to read

1. `git log --format='%h %an %ad %s' --date=short BASE..HEAD` and `git diff --stat BASE..HEAD`.
2. The diff of `PROJECT_HANDOFF.md` and `README.md`. These are the authoritative sources for milestones,
   conclusions, owners, and next steps.
3. Changed result files under `phase2/` (CSV or JSON) when the handoff cites them. Use them to confirm
   numbers, not to invent new analyses.
4. The current `docs/index.html`, so you change only what is needed.

## Where things live in docs/index.html

- `<script type="application/json" id="roadmap-seed">` near the top: roadmap tracks and plans shown
  when the file is opened directly. `status` is one of `idea`, `planned`, `active`, `done`.
- Hero: the headline, the `.sub` paragraph, and the `.byline` listing each track and owner.
- Section 一 (architecture): `.premise` lists what is fixed and what each track varies. `NODES` holds the
  component descriptions.
- Section 二 (tracks): the `SHARED` array (shared milestones) and the `TRACKS` array (one object per track,
  with `code`, `name`, `owner`, `q` and milestones `ms` of `{ s: "done"|"active"|"paused"|"next", t, r, f }`).
  A new track needs an entry in `TRACKS`, a line in `.byline`, and a `.premise` column if it adds a new
  variable. Track colors come from CSS rules like `.track[data-code="C"]`.
- Section 三 (results): chart data in `A_TYPES`, `B_ARMS`, `B_EXTRA`, `C_ROWS`, `D_X`, `D_SER`, and the
  prose in each `.take` block. Add a new results tab only when a track reports a finished result table.
- Section 四 (roadmap): rendered from `roadmap-seed`; update plans when the handoff changes next steps.
- Footer `.colophon`: the source and update date.

## Rules

- Every number and claim must come from the repository at HEAD. If the handoff qualifies a conclusion,
  use the same qualification. Never round a confidence interval into a stronger claim.
- Mark a milestone `done` only when the handoff reports its result. Use `next` for plans, `paused` for
  stopped runs, and `active` for work the handoff says is under way.
- Write in Traditional Chinese, in the page's existing voice: short, plain sentences; numbers with units.
- Keep the design. Do not change CSS, layout, fonts, colors, or script logic, except to add a color rule
  for a new track code or a new results tab that follows the existing pattern.
- Update the footer date to the date of the newest commit in the range, and the progress date in
  section 二 if milestones changed.
- Keep the JavaScript valid: balanced brackets, no trailing syntax errors, strings properly quoted.
- If nothing in the range affects the page (for example, only code or logs changed), leave the file
  unchanged. No edit means no pull request.

When you finish, reply with a short summary of what you changed on the page and which commits it reflects.
