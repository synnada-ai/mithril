# Setting up self-hosted runners

1. **Activate runner in github**
   - Follow the steps in [here](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/adding-self-hosted-runners) to enable CI runner in repository (or organization).
  
2. **Create run script that will work upon system login**
   1. create `.sh` file with following script
   ```bash
    #!/bin/bash
    source ~/.zprofile
    source ~/.zshrc
    bash /Path/to/Github-sh-file/run.sh
   ```

3. **Set up necessary mac-mini system settings**
   1. Click apple icon at the top left
   2. go to `System Settings`
   3. go to `Energy`
   4. Enable `Wake for network access`
   5. Enable `Start up automatically after a power failure`
   6. go to `Users & Gruops` on the left bar
   7. Enable auto login
   
4. **Create and initialize Daemon**
   1. go to `~/Library/LaunchAgents`
   2. create `com.user.startup.plist` that will include following script
   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
    <plist version="1.0">
    <dict>
        <key>Label</key>
        <string>com.user.startup</string>

        <key>Program</key>
        <string>/path/to/created/sh/file</string>

        <key>RunAtLoad</key>
        <true/>

        <key>KeepAlive</key>
        <true/>

        <key>StandardOutPath</key>
        <string>/tmp/startup.log</string>

        <key>StandardErrorPath</key>
        <string>/tmp/startup_error.log</string>
    </dict>
    </plist>
    ```

    - `Label`: defines label for the daemon
    - `Program`: script that will be run. In this case, `.sh` file written in step 2.
    - `RunAtLoad`: defines that script will be run upon user login.
    - `KeepAlive`: Program will be runned again in case of a failure.
    - `StandardOutPath`: path that stdout of the program will be written. Useful in debugging.
    - `StandardErrorPath`: path that error mesages of the program will be written. Useful in debugging.
  
   3. Ensure power permissions

    ```sh
        chmod 644 ~/Library/LaunchAgents/com.user.startup.plist
        chmod +x /path/to/created/sh/file.sh
    ```
   4. Launch the plist file

    ```bash
        launchctl load ~/Library/LaunchAgents/com.user.startup.plist
    ```


