# The README's demo: boxfit's output on the agent example, recorded output-only (the command
# runs hidden). Re-record from the repo root after any change to the output: vhs assets/demo.tape
Output assets/demo.gif
Set Shell "bash"
Set FontSize 14
Set Width 1560
Set Height 430
Set Padding 20
Set Theme "GitHub Dark"
Hide
Type `export PS1="" && python3 boxfit.py examples/gpt-oss-120b.config.json examples/gpt-oss-120b-agent.log --requests-per-day 200 > /tmp/bf-demo.txt && clear && sleep 1 && while IFS= read -r l; do printf "%s\n" "$l"; sleep 0.12; done < /tmp/bf-demo.txt`
Enter
Show
Sleep 9s
