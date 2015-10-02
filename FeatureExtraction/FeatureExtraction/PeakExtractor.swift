//  Copyright © 2015 Venture Media. All rights reserved.

import Foundation

import Surge

public typealias Point = Surge.Point<Double>

let yCutoff = 0.007
let minimumNoteDistance = 0.5

public class PeakExtractor {
    class public func process(input: [Point]) -> [Point] {
        let peaks = findPeaks(input)
        return filterPeaks(peaks)
    }

    class func findPeaks(input: [Point]) -> [Point] {
        var peaks = [Point]()
        
        for i in 1...input.count-2 {
            if input[i-1].y <= input[i].y && input[i].y >= input[i+1].y {
                peaks.append(input[i])
            }
        }
        
        return peaks
    }

    class func filterPeaks(input: [Point]) -> [Point] {
        let peaks = filterPeaksByHeight(input)
        return choosePeaks(peaks)
    }

    class func filterPeaksByHeight(input: [Point]) -> [Point] {
        return input.filter { (peak: Point) -> Bool in
            return peak.y > yCutoff
        }
    }

    class func choosePeaks(input: [Point]) -> [Point] {
        var chosenPeaks = [Point]()
        
        var currentPeakRange = Interval(min: 0, max: 0)
        for peak in input {
            if currentPeakRange.contains(peak.x) {
                if let lastPeak = chosenPeaks.last where lastPeak.y < peak.y {
                    chosenPeaks.removeLast()
                    chosenPeaks.append(peak)
                    currentPeakRange = binCutoffRange(peak.x)
                }
            } else {
                chosenPeaks.append(peak)
                currentPeakRange = binCutoffRange(peak.x)
            }
        }
        
        return chosenPeaks
    }

    class func binCutoffRange(freq: Double) -> Interval {
        let note = freqToNote(freq)
        
        let upperBound = noteToFreq(note + minimumNoteDistance)
        let lowerBound = noteToFreq(note - minimumNoteDistance)
        
        return Interval(min: lowerBound, max: upperBound)
    }
}
