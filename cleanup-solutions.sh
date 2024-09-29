#! /usr/bin/env bash
#
# This script is used to cleanup the solutions written in between blocks # BEGIN SOLUTION and # END SOLUTION.


# Find all the files in the current directory and its subdirectories
# that contain the pattern # BEGIN SOLUTION
# and store them in a variable
files=$(grep -rl "# BEGIN SOLUTION" **/*.py)

# Iterate over the files
for file in $files
do
    # Print the file name
    echo "Cleaning up $file"

    # Use sed to remove the lines between the patterns # BEGIN SOLUTION and # END SOLUTION
    gsed -i '/# BEGIN SOLUTION/,/# END SOLUTION/d' $file
done
