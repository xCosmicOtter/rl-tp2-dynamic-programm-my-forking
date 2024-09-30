#! /usr/bin/env bash
#
# This script is used to clean up the content between # BEGIN SOLUTION and # END SOLUTION without deleting the markers.
    # Use gsed to remove the lines between the patterns # BEGIN SOLUTION and # END SOLUTION but keep the markers
    # The below command deletes only the lines between the markers and leaves the markers themselves
    gsed -i '/# BEGIN SOLUTION/,/# END SOLUTION/{//!d;}' "$file"
