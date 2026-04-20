# Claude Rules for TeamBowl

## Plan Mode

Always start every task in plan mode. Before writing or editing any code, create a plan and get explicit approval from the user. Do not begin implementation until the user has approved the plan via ExitPlanMode.

## Allowed Shell Commands (no permission prompt needed)

`ls` and `find` (without `-exec`) are read-only and may always be run without asking for permission. Neither command can modify, move, delete, or write any file.

## 2026-04-19 — Fixed get_position() regex in test_lid.sh

`test_lid.sh` `get_position()` and `m` monitor command used `re.findall(r"'([^']+)'", data)`
to parse joint names from `ros2 topic echo /joint_states` output. ROS2 YAML output uses
`- joint_name` (no quotes), so `names` was always empty → always "NOT FOUND". Fixed to use
`re.findall(r'^-\s+(\S+)', name_section, re.MULTILINE)` in both locations.

## CLAUDE.md Maintenance

After every code change, update the `CLAUDE.md` file that is closest (nearest ancestor directory) to the file(s) you edited. Add or update a section that briefly describes what was changed, why, and any important context a future session would need. Keep these entries concise. If no `CLAUDE.md` exists in an ancestor directory other than the root, create one in the same directory as the file being edited.
