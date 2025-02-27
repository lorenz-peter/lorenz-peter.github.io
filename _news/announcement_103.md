---
layout: post
date: 2023-08-26 09:00:00-0400
inline: true
---

Check out my [writeups](https://lorenz-peter.github.io/blog/2023/gandalf) from the Lakera Gandalf hackathon.


("adil" "adrian" "bishwajit" "hans" "haolin" "jiayi" "minghui" "peter" "thomas" "trevor" "tristan" "zimeng")



ULIST=("adil" "adrian" "bishwajit" "hans" "haolin" "jiayi" "minghui" "peter" "thomas" "trevor" "tristan" "zimeng")

# Loop over each user
for USER in "${ULIST[@]}"; do
  # Check if .cache is already a symbolic link
  if [ ! -L /home/$USER/.cache ]; then
    echo $USER
    # Move the .cache folder to /hdd/$USER/.cache
    mkdir /hdd/$USER
    mv /home/$USER/.cache /hdd/$USER/.cache
    
    # Create a symbolic link to the original location
    ln -s /hdd/$USER/.cache /home/$USER/.cache
  fi