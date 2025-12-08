## Use
First Terminal
```bash
ssh ip@10.105.1.167
geicar
goto_jetson
cd AI/new_combined/
source venv/bin/activate
python3 main.py
```
—----------
New terminal
```bash
ssh ip@10.105.1.167
geicar
cd ros2_ws
source_ws
ros2 run pkg_tcp_receiver tcp_receiver
```
—----------
New Terminal
```bash
ssh ip@10.105.1.167
geicar
cd ros2_ws
source_ws
ros2 topic echo /ai_perception_data
```
