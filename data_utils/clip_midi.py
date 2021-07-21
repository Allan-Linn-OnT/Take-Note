import mido
import argparse
# Clean and clip the midi file by running it through mido
def clip_mid_file(file_location):
    '''
    file_location - path to file, we clip the file and save back to original location
    '''
    m = mido.MidiFile(file_location, clip=True)
    seen = {'time_signature':0, "set_tempo":0}
    for track in m.tracks:
        to_remove = []
        for i, mes in enumerate(track):
            if mes.is_meta:
                if mes.type in seen:
                    if seen[mes.type] == 0:
                        seen[mes.type]  += 1
                    else:
                        to_remove.append(i)
        for i in reversed(to_remove):
            track.pop(i)

    m.save(file_location)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help="mid file to clip")
    args = parser.parse_args()
    clip_mid_file(args.file)
