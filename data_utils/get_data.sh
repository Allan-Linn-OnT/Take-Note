#!/usr/bin/env bash

mkdir ../data
find ../data/* -print0 | xargs -0 rm -rf # make sure the data directory is empty, don't save anything in there

wget http://www-etud.iro.umontreal.ca/~boulanni/MuseData.zip
wget http://www-etud.iro.umontreal.ca/~boulanni/Piano-midi.de.zip
wget http://www-etud.iro.umontreal.ca/~boulanni/Nottingham.zip
wget http://www-etud.iro.umontreal.ca/~boulanni/JSB%20Chorales.zip

unzip MuseData.zip
unzip Piano-midi.de.zip
unzip Nottingham.zip
unzip "JSB Chorales.zip"

mv MuseData ../data
mv Piano-midi.de ../data
mv Nottingham ../data
mv "JSB Chorales" ../data

rm -rf *.zip

# need to do some processing of MuseData, and Piano-midi.de
dirs=("MuseData" "Piano-midi.de" "Nottingham")
echo $dirs

for d in "${dirs[@]}"
do
   echo "correcting $d"
   for midi in $( find ../data/$d -name "*.mid" )
        do
            xxd -p $midi | sed 's/\(ff5405.\{8\}\)../\100/' | xxd -r -p > $( dirname $midi )/temp # sed magic
            mv $( dirname $midi )/temp $midi
            rm -f $( dirname $midi )/temp

        done
done
