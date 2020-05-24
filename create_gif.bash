# Taken from following threads:
# https://superuser.com/questions/556029/how-do-i-convert-a-video-to-gif-using-ffmpeg-with-reasonable-quality
# https://unix.stackexchange.com/questions/24014/creating-a-gif-animation-from-png-files
# https://superuser.com/questions/714079/ffmpeg-slideshow-from-images-only-one-frame-shown

ffmpeg \
-pattern_type glob \
-i "output/depthmap_*.png" \
-vf "setpts=7*PTS, scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
output.gif

#-loop 0 \
#-ss 30 \
#-t 3 \
#-i output/depthmap_%02d.png \
