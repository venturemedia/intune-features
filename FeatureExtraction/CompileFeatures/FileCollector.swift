//  Copyright © 2015 Venture Media. All rights reserved.

import Foundation

class FileCollector {
    struct Example {
        var filePath: String
        var label: Int
    }
    
    let rootPath = "../AudioData/Notes/"
    let folders = [
        "AcousticGrandPiano_YDP",
        "Arachno",
        "FFNotes",
        "MFNotes",
        "PPNotes",
        "FluidR3_GM2-2",
        "GeneralUser_GS_MuseScore_v1.442",
        "Piano_Rhodes_73",
        "Piano_Yamaha_DX7",
        "TimGM6mb",
        "VenturePiano"
    ]
    
    let noteRange: Range<Int>
    let fileType: String
    let labelFunction: Int -> Int
    
    init(noteRange: Range<Int>, fileType: String, labelFunction: Int -> Int = { $0 }) {
        self.fileType = fileType
        self.noteRange = noteRange
        self.labelFunction = labelFunction
    }
    
    func buildExamples() -> [Example] {
        print("Working Directory: \(NSFileManager.defaultManager().currentDirectoryPath)")
        var examples = [Example]()
        for folder in folders {
            let array = examplesInFolder(folder)
            examples.appendContentsOf(array)
        }
        return examples
    }
    
    func examplesInFolder(folder: String) -> [Example] {
        let fileManager = NSFileManager.defaultManager()
        var examples = [Example]()
        for i in noteRange {
            let fileName = "\(i).\(fileType)"
            let notePath = buildPathFromParts([rootPath, folder, fileName])
            guard fileManager.fileExistsAtPath(notePath) else {
                print("File not found \(notePath)")
                continue
            }
            examples.append(Example(filePath: notePath, label: labelFunction(i)))
        }
        return examples
    }
}
