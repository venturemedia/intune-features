//  Copyright © 2015 Venture Media. All rights reserved.

import Foundation

import AudioKit
import FeatureExtraction
import HDF5
import Surge

let sampleCount = 32768

let featureCompiler = FeatureCompiler(sampleCount: sampleCount)
featureCompiler.compileFeatures()