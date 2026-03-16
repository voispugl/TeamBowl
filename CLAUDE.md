# Claude Rules for TeamBowl

## Plan Mode

Always start every task in plan mode. Before writing or editing any code, create a plan and get explicit approval from the user. Do not begin implementation until the user has approved the plan via ExitPlanMode.

## CLAUDE.md Maintenance

After every code change, update the `CLAUDE.md` file that is closest (nearest ancestor directory) to the file(s) you edited. Add or update a section that briefly describes what was changed, why, and any important context a future session would need. Keep these entries concise. If no `CLAUDE.md` exists in an ancestor directory other than the root, create one in the same directory as the file being edited.
