//
//  main.swift
//  CompileFeatures
//
//  Created by Aidan Gomez on 2015-10-01.
//  Copyright © 2015 Venture Media. All rights reserved.
//

import Foundation

import FeatureExtraction
import Surge
import AudioKit

typealias Point = Surge.Point<Double>

// Constant Parameters
let fileListPath = "file.txt"
let rootPath = ""
let sampleRate = 44100
let sampleCount = 32768

let numPeaks = 10
let numNotes = 88
let startNote = 24

// Computed Constants
let fft = FFT(inputLength: sampleCount)
let fb = Double(sampleRate) / Double(sampleCount)

// Objects
struct FeatureSet {
    var peaks = [Point](count: numPeaks, repeatedValue: Point()) {
        willSet {
            precondition(newValue.count == numPeaks)
        }
    }
    var meanSquare = [Double](count: numNotes, repeatedValue: 0.0){
        willSet {
            precondition(newValue.count == numNotes)
        }
    }
    var correlation = [Double](count: sampleCount, repeatedValue: 0.0){
        willSet {
            precondition(newValue.count == sampleCount)
        }
    }
}

// Read in filelist
let data = NSData(contentsOfFile: fileListPath)
guard let string = String(data: data!, encoding: NSUTF8StringEncoding) else {
    fatalError()
}

let splitLines = string.characters.split{ $0 == "\n" }
let lines = splitLines.map{ return String($0) }

// Process each file
var features = [FeatureSet]()
for fileName in lines {
    var featureSet = FeatureSet()
    
    let filePath = rootPath + fileName
    let audioFile = AudioFile(filePath: filePath)!
    assert(audioFile.sampleRate == 44100)
    
    var data = [Double](count: sampleCount, repeatedValue: 0.0)
    audioFile.readFrames(&data, count: sampleCount)
    
    let psd = sqrt(fft.forwardMags(data))
    
    let fftPoints = (0..<psd.count).map{ Point(x: fb * Double($0), y: psd[$0]) }
    let peaks = process(fftPoints).sort{ $0.y > $1.y }
    featureSet.peaks = Array(peaks[0..<numPeaks])
    
    let correlationData = autocorrelation(data)
    featureSet.correlation = correlationData
    
    var meanSquareData = [Double]()
    for note in startNote..<(startNote + numNotes) {
        let noteLowerBound = Int(floor(noteToFreq(Double(note) - 0.5) / fb))
        let noteUpperBound = Int(ceil(noteToFreq(Double(note) + 0.5) / fb))
        
        let slice = Array(psd[noteLowerBound...noteUpperBound])
        meanSquareData.append(measq(slice))
    }
    featureSet.meanSquare = meanSquareData
    
    features.append(featureSet)
}