cmd_/home/box/teambowl_tools/xsens_mt/xsens_mt.mod := printf '%s\n'   xsens_mt.o | awk '!x[$$0]++ { print("/home/box/teambowl_tools/xsens_mt/"$$0) }' > /home/box/teambowl_tools/xsens_mt/xsens_mt.mod
