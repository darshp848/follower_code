# Taking the arms from a Dream station and giving them back

The station agent (`robots.station`) holds both CAN links and `~/.dreamlabs/arms.lock`. `leader_follower.py` needs
the CAN links to itself, so the station has to be stopped while it runs.

1. **Check the station is idle.** If `curl -s localhost:8091/status` reports `"state": "running"`, someone's run is in
   progress. Wait for it to finish.
2. **Note how the station was started.** Read the start options from `dreamcli robot log`, the station log, or the
   person who started it (`robot up`, `robot up nosys0` or `robot up joint`). You need them in step 7.
3. **Home the arms, and stand at them.** `dreamcli robot down` stops the agent with SIGINT, and the arms switch off and
   go limp. Support the arms, or make sure they're resting, before you run it.
4. `dreamcli robot down`
5. **Copy the script to the station and run it.** Use the station's Python, which already has i2rt and OpenCV:

   ```bash
   ~/dream-env/bin/python leader_follower.py --leader can_follower_l --follower can_follower_r \
       --cam scene=<scene device> --cam wrist_r=<wrist_r device>
   ```

   Camera device paths are under `cameras:` in `~/.dreamlabs/station.yaml`.
6. **Quit with `q`.** Both arms go back to their start-up pose before the buses close.
7. **Restart the station** with the same options as before, e.g. `dreamcli robot up`, then check that
   `curl -s localhost:8091/status` reports both arms.
