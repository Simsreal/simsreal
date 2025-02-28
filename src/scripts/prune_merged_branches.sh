#!/bin/bash

# Fetch and prune
git fetch --all --prune

# Get the current branch
current_branch=$(git symbolic-ref --short HEAD)

# Get local branches not on remote, excluding the current branch
branches_to_delete=$(comm -23 <(git branch | sed 's/^\* //' | grep -v "^$current_branch$" | sort) <(git branch -r | sed 's/origin\///' | sort))

# Delete each branch
for branch in $branches_to_delete; do
    echo "Deleting branch: $branch"
    git branch -d $branch
done
