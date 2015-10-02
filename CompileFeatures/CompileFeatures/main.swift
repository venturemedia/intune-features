//  Copyright © 2015 Venture Media. All rights reserved.

import Foundation

import AudioKit
import FeatureExtraction
import HDF5
import Surge

typealias Point = Surge.Point<Double>

// Constant Parameters
let fileListPath = "file.txt"
let rootPath = ""
let ratio = 10

let sampleRate = 44100
let sampleCount = 32768

let numPeaks = 10
let numNotes = 88
let startNote = 24

// Computed Constants
let fft = FFT(inputLength: sampleCount)
let fb = Double(sampleRate) / Double(sampleCount)
let dataShape: [UInt64] = [UInt64(numPeaks + numNotes + sampleCount)]
let labelShape: [UInt64]

// Objects
struct FeatureSet {
    var label: Int = 0
    var peaks = [Point](count: numPeaks, repeatedValue: Point()) {
        willSet {
            precondition(newValue.count == numPeaks)
        }
    }
    var bands = [Double](count: numNotes, repeatedValue: 0.0){
        willSet {
            precondition(newValue.count == numNotes)
        }
    }
    var correlation = [Double](count: sampleCount, repeatedValue: 0.0){
        willSet {
            precondition(newValue.count == sampleCount)
        }
    }
    func data() -> [Double] {
        var data = [Double]()
        for point in peaks {
            data.appendContentsOf([point.x, point.y])
        }
        data.appendContentsOf(bands)
        data.appendContentsOf(correlation)
        
        return data
    }
}

// Read in filelist
let data = NSData(contentsOfFile: fileListPath)
guard let string = String(data: data!, encoding: NSUTF8StringEncoding) else {
    fatalError()
}

let lines = string.characters.split{ $0 == "\n" }
let splitLines = lines.map{ return $0.split{ $0 == " " } }
var dataLabelPairs = [(String, Int)]()
for line in splitLines {
    let dataFileName = String(line[0])
    guard let label = Int(String(line[1])) else {
        fatalError("Invalid label: could not be converted to Int")
    }
    dataLabelPairs.append((dataFileName, label))
}
labelShape = [UInt64(lines.count)]

// Process each file
var features = [FeatureSet]()
for (fileName, label) in dataLabelPairs {
    var featureSet = FeatureSet()
    featureSet.label = label
    
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
    
    var bands = [Double]()
    for note in startNote..<(startNote + numNotes) {
        let noteLowerBound = Int(floor(noteToFreq(Double(note) - 0.5) / fb))
        let noteUpperBound = Int(ceil(noteToFreq(Double(note) + 0.5) / fb))
        
        let slice = Array(psd[noteLowerBound...noteUpperBound])
        bands.append(sum(slice))
    }
    featureSet.bands = bands
    
    features.append(featureSet)
}

// Create database and write data
let split = features.count/ratio
let testingFeatures = features[0..<split]
let trainingFeatures = features[split...features.count]
let databaseNameAndData = [("training", trainingFeatures), ("testing", testingFeatures)]

let dataType = HDF5.Datatype.copy(type: .Double)
let labelType = HDF5.Datatype.copy(type: .Int)

for (name, data) in databaseNameAndData {
    guard let hdf5File = HDF5.File.create(name, mode: File.CreateMode.Truncate) else {
        fatalError("Could not create HDF5 dataset.")
    }

    let hdf5DataDataspace = Dataspace(dims: dataShape)
    let hdf5Data = HDF5.Dataset.create(file: hdf5File, name: "data", datatype: dataType, dataspace: hdf5DataDataspace)
    
    let hdf5LabelDataspace = Dataspace(dims: labelShape)
    let hdf5Labels = HDF5.Dataset.create(file: hdf5File, name: "labels", datatype: labelType, dataspace: hdf5LabelDataspace)
    
    for featureSet in data {
        assert(hdf5Labels.writeInt([featureSet.label]))
        assert(hdf5Data.writeDouble(featureSet.data()))
    }
}
