//  Copyright © 2015 Venture Media. All rights reserved.

import Foundation

import Surge

public class BandExtractor {
    class public func process(spectrumData data: [Double], startNote: Int, numberOfNotes num: Int, baseFrequency fb: Double) -> [Double] {
        var bands = [Double]()
        for note in startNote..<(startNote + num) {
            let noteLowerBound = Int(floor(noteToFreq(Double(note) - 0.5) / fb))
            let noteUpperBound = Int(floor(noteToFreq(Double(note) + 0.5) / fb))
            
            let slice = Array(data[noteLowerBound...noteUpperBound])
            bands.append(sum(slice))
        }
        return bands
    }
}
