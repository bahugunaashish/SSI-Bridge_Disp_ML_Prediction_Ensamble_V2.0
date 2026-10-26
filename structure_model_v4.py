Warning: truncated output (original token count: 27189)
Total output lines: 1800

# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import logging
import sys
import os
import openseespy.opensees as op
import opsvis as opsv
import matplotlib.pyplot as plt
import math
import numpy as np
import time as tt
from matplotlib.animation import FuncAnimation
from units_v4 import get_units, CreateEqDof, CollectMaxNodes
from getMat_v4 import *
from getSection2_v4 import *

from config_inputs_v4 import *
from helpers_v4 import rotate_node, transverse_capbeam_locations


def _soil_layer_for_model_elevation(model_z, soil_layers, profile_to_model_offset, tolerance=1.0e-8):
    """Find the profile layer, extending the bottom layer below profile elevation 0 m."""
    profile_z = float(model_z) - float(profile_to_model_offset)
    if profile_z < 0.0 - tolerance:
        return soil_layers[-1], profile_z, True

    for layer_index, layer in enumerate(soil_layers):
        is_last_layer = layer_index == len(soil_layers) - 1
        at_profile_bottom = is_last_layer and math.isclose(
            profile_z, float(layer['bottom']), rel_tol=0.0, abs_tol=tolerance
        )
        if float(layer['top']) >= profile_z > float(layer['bottom']) or at_profile_bottom:
            return layer, profile_z, False
    return None, profile_z, False


#%% Structure Model Class
class Define_str:
    def __init__(self, L1, AllSoilNodes, SocketLength, SocketElementType, xlocation, ylocation, SecTag, ShaftLength, 
                   ShaftElementType, ColumnLength, ColumnElementType, bearingLocation, CapBeamElementType, Depth_CapBeam, HPileLength, pileLocation,
                   soil_profile=None, load_data=None, geometry=None):
    # def __init__(self, SocketLength, SocketElementType, xlocation, ylocation, SecTag, ShaftLength, 
                  # ShaftElementType, ColumnLength, ColumnElementType, bearingLocation, CapBeamElementType, Depth_CapBeam, HPileLength, pileLocation):
        global m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi, degrees
        global gamma_rc, gamma_c, gamma_s, Ubig, Usmall, g
        # m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi, lbf, pcf, psf, inch2, inch4, g, Ubig, Usmall= get_units2()

        m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi = get_units()   
        #%% define variables
        self.theta = 0. # in degree
        self.L1               = L1
        self.AllSoilNodes     = AllSoilNodes
        self.soil_profile     = soil_profile
        self.load_data        = load_data
        self.geometry         = geometry or {}
        self.column_top_z     = self.geometry.get('column_top_z', 104*ft)
        self.column_bottom_z  = self.geometry.get('column_bottom_z', 98*ft)
        self.shaft_top_z      = self.geometry.get('shaft_top_z', 96*ft)
        self.shaft_bottom_z   = self.geometry.get('shaft_bottom_z', 58*ft)
        self.socket_top_z     = self.geometry.get('socket_top_z', 56*ft)
        self.socket_bottom_z  = self.geometry.get('socket_bottom_z', 24*ft)
        self.pile_top_z       = self.geometry.get('pile_top_z', 104*ft)
        self.pile_bottom_z    = self.geometry.get('pile_bottom_z', 48*ft)
        self.ground_surface_z = self.geometry.get('ground_surface_z', 102*ft)
        self.pile_diameter    = self.geometry.get('pile_diameter', 2*29.75*inch)
        loads = self.load_data or {}
        self.columnMass = loads.get('columnMass', columnMass)
        self.ShaftMass = loads.get('ShaftMass', ShaftMass)
        self.SocketMass = loads.get('SocketMass', SocketMass)
        self.PileMass = loads.get('PileMass', PileMass)
        # self.L2               = L2 # Soil defined 
        self.SocketLength       = SocketLength                                              # socket length
        self.SocketElementType  = SocketElementType                                         # socket element type
        self.xlocation          = xlocation                                                 # location of column in interior bent
        self.ylocation          = ylocation                                                 # cumulative span lengths
        self.EndBentLocation    = np.array([self.ylocation[0], self.ylocation[-1]])        # Location of end bent
        self.SecTag             = SecTag                                                    # Section Tags
        self.ShaftLength        = ShaftLength                                               # Shaft length
        self.ShaftElementType   = ShaftElementType                                          # shaft element type
        self.ColumnLength       = ColumnLength                                              # Column length
        self.ColumnElementType  = ColumnElementType                                         # Column element type
        self.CapBeamElementType = CapBeamElementType                                        # Cap beam element type
        self.PileElementType    = ColumnElementType
        self.SocketShaftLength  = self.ShaftLength + self.SocketLength      # Length of socket + shaft
        self.PierLength         = self.ColumnLength+self.SocketShaftLength  # Total length of pier
        self.bearingLocation    = bearingLocation                           # rigid element type
        self.capBeamLocations   = transverse_capbeam_locations(self.xlocation, self.bearingLocation)
        self.ColumnBeamRigidLnk = Depth_CapBeam/2                           # rigid link length at interior bent
        self.ColumnGeomTranType = 'PDelta'                                  # Column geometric transform type
        self.bearingdepth       = 15/4*inch                                    # bearingdepth
        self.BearingElementType = CapBeamElementType                        # Bearing element type
        self.DiaphElementType   = CapBeamElementType                        # Diaphragm element type
        self.RigidElementType   = CapBeamElementType                        # rigid element type
        self.HplieLength        = HPileLength                               # Pile length
        self.NumPiles           = int(len(pileLocation))                    # number of piles in a row
        self.EndCapBeamLength   = 616.5*inch                                # Length of end bent cap beam
        self.pileLocation       = pileLocation
        self.IntCapBeamDepth    = 4.5*ft                                    # depth of interior cap beam section
        self.topCapBeam         = self.PierLength + self.IntCapBeamDepth    # length from top of interior cap beam to bottom of socket
        self.EndCapBeamDepth    = 3.5*ft
        self.EffPileLength      = self.HplieLength-self.EndCapBeamDepth/2
        self.EndPierLength      = self.HplieLength+self.EndCapBeamDepth/2
        self.BottomPileNodes    = np.array([])
        self.fixNodesTags       = np.array([])
        self.AllDiaphTags       = np.array([])
        self.IntDiaphEleTag     = np.array([])
        self.EndDiaphEleTag     = np.array([])
        self.connection_warnings = []
        #%% Large values to provide rigid elements
        
        self.Ei = 40*GPa
        self.Ai = 9000000.0*m**2
        self.Ii = 900000000.0*m**4
        self.Ji = 900000000.0*m**4
        self.Gi = 900000.0*N/m**2
        
        self.RigidProp = [self.Ai, self.Ei, self.Gi, self.Ji, self.Ii, self.Ii]
        #defining mass
        
        # print(self.topCapBeam , self.EffPileLength+self.EndCapBeamDepth/2)
        #%% 1- Node Tags
        self.tagSo       = 1000        # Socket node tag identifier
        self.tagS        = 3000        # Shaft node tag identifier
        self.tagC        = 5000        # Column node tag identifier
        self.tagCB       = 7000        # Cap Beam node tag identifier
        self.rigidTag    = 9000        # rigid node tag
        self.BearingTag  = 14000        # Bearing node tag
        self.tagPile     = 16000
        self.tagECB      = 18000        # End Cap Beam node tag identifier
        self.tagD        = 0
        self.eqDofnodes  = 19000
        #%% 2- Element Tags  
        self.DeckEleTag       = 0
        self.SocketEleTag     = 1000        # Socket element tag identifier
        self.ShaftEleTag      = 3000        # Shaft element tag identifier
        self.ColumnEleTag     = 5000        # Column element tag identifier
        self.CapBeamEleTag    = 7000        # Cap Beam element tag identifier
        self.RigidEleTag      = 9000        # Rigid link element tag identifier
        self.BearingEleTag    = 40000        # Bearing element tag
        self.DiaphEleTag      = 41000        # Diaphragm element tag
        self.PileEleTag       = 42000
        self.ECapBeamEleTag   = 43000        # Cap Beam element tag identifier
        self.zeroEleTag       = 45000        # zerolength element tags  
        self.EpTag            = 48000        # earthpressure nodes
        self.zeroEleTags = np.array([], dtype=int) 
        
        self.geomTTag      = 0           # Geometrtic transformation tag 
        self.InteTag       = 0           # Beam Integration Tag identifer
        
        self.DeckTags = np.array([], dtype=int)     # array for all Deck Nodes
        self.GirderNodeTags = np.array([], dtype=int) # array for all Girder node Tags
        self.ColumnTopTags = np.array([], dtype=int)
        self.PileTopTags = np.array([], dtype=int)
        self.DeckEleTags = np.array([], dtype=int)
        self.SocketEleTags = np.array([], dtype=int)
        self.ShaftEleTags = np.array([], dtype=int)
        self.ColumnEleTags = np.array([], dtype=int)
        self.CapBeamEleTags =  np.array([], dtype=int)
        self.ECapBeamEleTags = np.array([], dtype=int)
        self.PileElemTags = np.array([], dtype=int)
        self.BottomNodesSo = np.array([], dtype=int)
        self.ShaftColumnNodeTags = np.array([], dtype=int)
        self.EndbentPileNodes = np.array([], dtype=int)
        self.zeroEleTags = np.array([], dtype=int)
        self.EndbentNodeTags = np.array([], dtype=int)
        self.EndbentElemTags = np.array([], dtype=int)
        #%%call all major function (Methods)
        self.initialize_model()
        self._buildStructure_()
        
        # self._Connect_Soil_Pile()
        self._JoinSoilPile()

    def _warn_unmatched_capbeam_nodes(self, source_nodes, capbeam_nodes, label):
        """Warn when a pile/column top has no matching cap-beam node."""
        tolerance = 1.0e-6
        capbeam_x = [op.nodeCoord(int(node))[0] for node in capbeam_nodes]
        unmatched = []
        for source_node in source_nodes:
            source_x = op.nodeCoord(int(source_node))[0]
            if not any(np.isclose(source_x, cap_x, rtol=0.0, atol=tolerance) for cap_x in capbeam_x):
                unmatched.append(source_x)

        if unmatched:
            message = (
                f"{label}: {len(unmatched)} pile/column top node(s) have no "
                f"matching cap-beam node at x={unmatched}"
            )
            self.connection_warnings.append(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)

        return unmatched
        self._eleload()
        # if len(self.AllSoilNodes)==0:
          
            # self.boundary_()  # acitvate if only superstructre needs to analyzed
    
    def getPileElem(self):
        return self.SocketEleTags.tolist(), self.ShaftEleTags.tolist(), self.ColumnEleTags.tolist(), self.BottomNodesSo.tolist()
        
    def getPileNodes(self):
        
        return self.ShaftColumnNodeTags.tolist()
        
        #%%   Initalization ofmodel     
    def initialize_model(self):
        print('Structure model initialized >>')
        op.model('basic', '-ndm', 3, '-ndf', 6)
        
    #%% boundary for strcture without soil     
    def boundary_(self):
        for ii in self.BottomNodesSo:
            print('fix nodes', ii)
            op.fix(int(ii), 1,1,1,1,1,1)
        
    #%% Connecting soil and pile
    def _Connect_Soil_Pile(self):
                
        for L in self.L1:
            #------------------------------------------------------------------
            #NodeSet = L.create_new_nodes_around_holes()
            # NodeSet = L.getNodes()
            # newNodes = self.create_new_nodes_around_holes(NodeSet)            use this when holes are cretaed
            # print(newNodes)     
            #------------------------------------------------------------------
            print('Nodes to connect the pile and soil are generated...')
            # print('Shaft and Column nodes',self.ShaftColumnNodeTags)
            
            # self._JoinNodes(newNodes)  # use this when self.create_new_nodes_around_holes is used 
            self._JoinNodes() 
            
    #%% create_new_nodes_around_holes
    def create_new_nodes_around_holes(self, NodesSet):
        """
        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        op.model('basic', '-ndm', 3, '-ndf', 6)
        print('Creating new nodes around Piles')
        new_node_id = op.getNodeTags()[-1]+1
        new_nodes = []
        # print(f'new node id {self.node_id}')
        # print('soil nodes around pile',self.nodes_around_holes)
        for node_id in NodesSet:
            # print(node_id)
            x, y, z = op.nodeCoord(node_id)
            
            if z != 25*ft :
              op.node(new_node_id, x, y, z)
              new_nodes.append(new_node_id)
              op.equalDOF(new_node_id, node_id, 1,2,3)
              # print(self.new_node_id, node_id, x, y, z)
              # print(new_node_id, node_id)
              new_node_id += 1
        # print('New Nodes around Piles', self.new_nodes)      
        return new_nodes
    
   
                
    #%% joint node when soil and pile connected through rigid elements
    def _JoinNodes(self, NodeList):
        # print(f'Node List {NodeList}')
        
        nodeList = []
        # print(NodeList)
        for Snode in NodeList:                    # Node List should be All soil Nodes
            for Cnode in self.ShaftColumnNodeTags:
                Scord = op.nodeCoord(int(Snode))
                Ccord = op.nodeCoord(int(Cnode))
                
                if np.linalg.norm(Ccord[2] - Scord[2]) < 1e-6 and np.linalg.norm(Ccord[1] - Scord[1])<70*inch and np.linalg.norm(Ccord[0] - Scord[0])<70*inch:
                
                    # print('Connecting ', Snode, Scord, Cnode, Ccord)
                    xaxis = np.subtract(Ccord, Scord)
                    vecxy = [0,0,1] # What you want (vecxy is global vertical)
                    vecxz = np.cross(xaxis,vecxy) # What OpenSees expects
                    self.geomTTag = self.geomTTag+1
                    op.geomTransf('Linear', self.geomTTag, *vecxz)
                    self.RigidEleTag = self.RigidEleTag+1
                    op.element('elasticBeamColumn', int(self.RigidEleTag), int(Cnode), int(Snode),  *self.RigidProp, self.geomTTag)
                # elif np.linalg.norm(Ccord[1] - Scord[1])<50 and np.linalg.norm(Ccord[0] - Scord[0])<50:
                #     # print('Not Connecting ', Snode, Scord, Cnode, Ccord)
                #     # sdf =  [Snode, Scord, Cnode, Ccord]
                #     nodeList.append([Snode, Scord, Cnode, Ccord])
                else:
                    
                    nodeList.append([Snode, Scord, Cnode, Ccord])
                    # print('not Connecting ', Snode, Scord, Cnode, Ccord)
        output_file = 'NodeList.txt'
        # Save nodeList to a text file
        # with open(output_file, 'w') as file:
        #     for node in nodeList:
        #         file.write(f'{node[0]} {node[1]} {node[2]} {node[3]}\n')
                
        #Connect Bottom nodes of the drilled shaft
        print('Connecting Bottom nodes of Shaft to Soil nodes')
        BottomnodeList = []
        for nJp in self.BottomNodesSo:
            pcord = op.nodeCoord(int(nJp))
            
            for nIs in self.AllSoilNodes:
                scord = op.nodeCoord(int(nIs))
                if np.linalg.norm(pcord[2] - scord[2]) <= 1e-5 and np.linalg.norm(pcord[1] - scord[1])<=1e-5 and np.linalg.norm(pcord[0] - scord[0])<=1e-5:
                    # op.equalDOF(int(nIs),int(nJp), 1 , 2, 3)
                    op.fix(int(nJp), 1, 1, 1, 1, 1, 1)
                    print( 'pile and soil node at the bottom',nJp, pcord, nIs, scord) 
                else:
                    # print(nJp, pcord, nIs, scord)
                    BottomnodeList.append([nJp, pcord, nIs, scord])
    
        output_file = 'BottomNodeList.txt'
        
        # Save nodeList to a text file
        # with open(output_file, 'w') as file:
        #     for node in BottomnodeList:
        #         file.write(f'{node[0]} {node[1]} {node[2]} {node[3]}\n')      
    
        return  BottomnodeList#, nodeList              
                    # RigidElements = np.append(RigidElements, RigidElementid)
        
       
    #%% strcutre building
    def _buildStructure_(self):
        print('modelling started...')
        if len(self.ylocation)>=2:
            print(f'number of bents: {len(self.ylocation)}')
            for ylocation in range (len(self.ylocation)):
                self.SocketLastTags  = np.array([])  
                self.ShaftLastTags   = np.array([])
                self.ColumnLastTags  = np.array([])
                self.CapBeamLastTags = np.array([])
                    
                    #%% create Socket nodes and elements in transverse direction
                    # condition to check for interior bent
                 # condition to check for interior bent
                if self.ylocation[ylocation] != self.ylocation[0] and self.ylocation[ylocation] != self.ylocation[-1]:    
                    #print(self.ylocation[ylocation],'True')
                    
                    # # Create the nodes
                    for Clocation in range (len(self.xlocation)):
                        # print(self.xlocation[Clocation], self.ylocation[ylocation])
                        Col_xT, Col_yT = rotate_node(self.xlocation[Clocation], self.ylocation[ylocation],  self.theta)
                        top_coord = [Col_xT, Col_yT, self.socket_top_z]
                        bottom_coord = [Col_xT, Col_yT, self.socket_bottom_z]
                        # top_coord = [self.xlocation[Clocation], self.ylocation[ylocation], 56*ft] #58ft
                        # bottom_coord = [self.xlocation[Clocation], self.ylocation[ylocation], 24*ft] #25ft
                        total_length = top_coord[2] - bottom_coord[2]  # Since Z-coordinates are negative for depth
                        # print(total_length)
                        # # Calculate the spacing between each node
                        self.num_elements = 16
                        spacing = total_length / self.num_elements
            
                        # # Generate the coordinates for each node
                        z_coords = [bottom_coord[2] + i * spacing for i in range(self.num_elements + 1)]
                        # print('Length of ',len(z_coords))
                        self.Sotag1 = np.array([])
                        SocketNodeMass = self.SocketMass/(self.num_elements+1)
                        for z in z_coords:
                            self.tagSo += 1
                            self.Sotag1 = np.append(self.Sotag1, self.tagSo ) 
                            self.ShaftColumnNodeTags = np.append(self.ShaftColumnNodeTags, self.tagSo )
                            # print(f'Node ID: {self.tagSo} Coordinates: ({Clocation,ylocation,z})')


                            op.node(int(self.tagSo), Col_xT, Col_yT, z, '-mass', SocketNodeMass, SocketNodeMass, SocketNodeMass, 0., 0., 0.) # with roation
                            # op.node(int(self.tagSo), self.xlocation[Clocation], self.ylocation[ylocation], z, '-mass', SocketNodeMass, SocketNodeMass, SocketNodeMass, 0., 0., 0.) # orginal
                            # print(self.tagSo, z)
                        self.SocketLastTags = np.append(self.SocketLastTags, self.tagSo)
                        # print('last node tag for socket',self.SocketLastTags)
                        self.BottomNodesSo = np.append(self.BottomNodesSo , self.Sotag1[0])
                        # print('Bottom node tag for socket',self.BottomNodesSo)
                        #creating geomteric transformation tag
                        self.geomTTag = self.geomTTag+1
                        
                        #creating geomteric transformation
                        op.geomTransf('PDelta', self.geomTTag, 1., 0., 0.)
                        
                        #creating Integration type
                        IntegrationType = 'Lobatto'
                        
                        #creating Integration tag
                        self.InteTag = self.InteTag+1
                        
                        #number of Intergation points
                        NumIP = 5
                        
                        #creating Integration 
                        op.beamIntegration(IntegrationType, self.InteTag, int(self.SecTag[4]), NumIP)
                        
                        self.SocketEleTagC = np.array([])
                        for nodes in range(len(self.Sotag1)-1):
                            
                            self.SocketEleTagC = np.append(self.SocketEleTagC,self.SocketEleTag)
                            # #element('dispBeamColumn', eleTag, *eleNodes, transfTag, integrationTag, '-cMass', '-mass', mass=0.0)
                            op.element(self.SocketElementType, int(self.SocketEleTag), int(self.Sotag1[nodes]), int(self.Sotag1[nodes+1]), self.geomTTag, self.InteTag)
                            self.SocketEleTags = np.append(self.SocketEleTags, self.SocketEleTag)
                            self.SocketEleTag += 1
                        # return  self.SocketEleTags, self.SocketLastTags  
                        
                    #%% Shaft nodes and elements   
                    for Clocation in range (len(self.xlocation)):
                        Col_xT, Col_yT = rotate_node(self.xlocation[Clocation], self.ylocation[ylocation],  self.theta)
                        top_coord = [Col_xT, Col_yT, self.shaft_top_z]
                        bottom_coord = [Col_xT, Col_yT, self.shaft_bottom_z]
                        # print(self.xlocation[Clocation], self.ylocation[ylocation])
                        # top_coord = [self.xlocation[Clocation], self.ylocation[ylocation], 96*ft]
                        # bottom_coord = [self.xlocation[Clocation], self.ylocation[ylocation], 58*ft]  #59
                        total_length = top_coord[2] - bottom_coord[2]  # Since Z-coordinates are negative for depth
                        # print(total_length)
                        # # Calculate the spacing between each node
                        self.num_elements = 19 #37
                        self.spacing = total_length / self.num_elements
        
                        # # Generate the coordinates for each node
                        z_coords = [bottom_coord[2] + i * self.spacing for i in range(self.num_elements + 1)]
                        
                        self.Stag1 = np.array([])
                        ShaftNodeMass = self.ShaftMass/(self.num_elements+2)
                        #creating nodes with tags
                        
                        for z in z_coords:
                            self.tagS += 1
                            self.Stag1 = np.append(self.Stag1,self.tagS) #to store all nodes in one array
                            op.node(self.tagS, Col_xT, Col_yT, z, '-mass', ShaftNodeMass, ShaftNodeMass, ShaftNodeMass, 0., 0., 0.) # with rotation
                            # op.node(self.tagS, self.xlocation[Clocation], self.ylocation[ylocation], z, '-mass', ShaftNodeMass, ShaftNodeMass, ShaftNodeMass, 0., 0., 0.) # origin…15189 tokens truncated…', 0, 0., -loads["ShaftWt"])
      op.eleLoad('-ele', *self.ColumnEleTags,   '-type', '-beamUniform', 0, 0., -loads["columnWt"])
      op.eleLoad('-ele', *self.CapBeamEleTags,  '-type', '-beamUniform', -loads["CapBeamWt"], 0., 0.)
      op.eleLoad('-ele', *self.ECapBeamEleTags, '-type', '-beamUniform', -loads["ECapBeamWt"], 0., 0.)
      op.eleLoad('-ele', *self.PileElemTags,    '-type', '-beamUniform', 0., 0., -loads["PileWt"])
      op.eleLoad('-ele', *self.EndDiaphEleTag,  '-type', '-beamUniform', -loads["DiaPhragmLoad"], 0., 0.)
      op.eleLoad('-ele', *endbentdiaph,         '-type', '-beamUniform', -loads["AdditionalLoad"], 0., 0.)
    
      return self.DeckEleTags, self.SocketEleTags, self.ShaftEleTags, self.ColumnEleTags, self.CapBeamEleTags, self.ECapBeamEleTags, self.PileElemTags 
           

    def _JoinSoilPile(self):
        """
        Automated Beam-on-Nonlinear-Winkler-Foundation (BNWF).
        Generates fixed soil nodes at the exact coordinates of the pile nodes
        and connects them with zeroLength p-y, t-z, and q-z springs.
        Includes Parallel Viscous materials for Radiation Damping.
        """
              
        print('\n--- Building Winkler Foundation ---')
        
        # Initialize storage for the master spring plots
        self.spring_curves = {'py': [], 'tz': [], 'qz': []}
        
        # Initialize array to hold the base soil nodes for dynamic excitation
        self.BaseSoilNodes = np.array([], dtype=int)
        self.bottom_node_soil_assignments = []
        
        # Ensure units_v4 are defined in your global scope, or define them here if needed
     
        ground_surface_z = self.ground_surface_z
        bottom_of_cap_z = 102.0 * ft  # No soil springs at or above this elevation
        pile_diameter = self.pile_diameter
        
        # Tag counters
        soil_node_tag = 80000 
        mat_tag_base  = 100000 
        
        col_coords = {int(n): tuple(op.nodeCoord(int(n))) for n in self.ShaftColumnNodeTags}

        # Keep bottom-node checks tag based. BottomNodesSo contains both the
        # interior socket bases and the end H-pile tips; these belong to
        # different tag ranges, so do not infer tips from a single prefix.
        bottom_node_tags = {int(n) for n in self.BottomNodesSo}
        missing_bottom_nodes = bottom_node_tags.difference(col_coords)
        if missing_bottom_nodes:
            raise RuntimeError(
                "Bottom node tags are missing from ShaftColumnNodeTags: "
                f"{sorted(missing_bottom_nodes)}"
            )

        active_soil_profile = (
            self.soil_profile if self.soil_profile is not None
            else global_i95_soil_profile
        )
        if not active_soil_profile:
            raise ValueError("No soil profile is available for pile-soil spring generation")

        # Soil profile elevations use a local datum. Align the profile surface
        # with the model ground surface, then compare actual node elevations
        # in that common coordinate system.
        profile_surface_z = float(active_soil_profile[0]['top'])
        profile_bottom_z = float(active_soil_profile[-1]['bottom'])
        profile_to_model_offset = ground_surface_z - profile_surface_z
        # The soil profile datum is 0 m at its bottom. Keep model node
        # elevations unchanged and translate them into that local datum.
        model_profile_bottom_z = profile_to_model_offset
        profile_tolerance = 1.0e-8
        bottom_nodes_below_profile = [
            (node_tag, float(col_coords[node_tag][2]))
            for node_tag in sorted(bottom_node_tags)
            if float(col_coords[node_tag][2]) < model_profile_bottom_z - profile_tolerance
        ]
        if bottom_nodes_below_profile:
            warnings.warn(
                "Bottom nodes extend below the soil profile; the deepest soil "
                "layer properties will be used for those nodes. "
                f"Model soil-bottom elevation={model_profile_bottom_z:.9g}; "
                f"node tags/elevations={bottom_nodes_below_profile}",
                RuntimeWarning,
                stacklevel=2,
            )

        bottom_tz_nodes = set()
        bottom_qz_nodes = set()
      
        
        # Group nodes into individual piles based on (X, Y)
        piles = {}
        for node, (x, y, z) in col_coords.items():
            loc_key = (round(x, 4), round(y, 4))
            # x, y = round(x, 1), round(y, 1)  # ← add this one line
            # loc_key = (x, y)   
            if loc_key not in piles:
                piles[loc_key] = {}
            piles[loc_key][node] = (x, y, z)

        # Loop through each pile individually
        for loc_key, pile_nodes in piles.items():
            
            # Sort ONLY this specific pile's nodes by depth (highest elevation to lowest)
            sorted_pile_nodes = sorted(pile_nodes.items(), key=lambda item: item[1][2], reverse=True)
            
            for i, (Cnode, (x, y, z)) in enumerate(sorted_pile_nodes):
                
                # Skip nodes above the groudn surfcae
                if z >= bottom_of_cap_z:
                    continue
                    
                # 1. CREATE THE FIXED SOIL NODE
                Snode = soil_node_tag
                op.node(Snode, x, y, z)
                op.fix(Snode, 1, 1, 1, 1, 1, 1) 
                # print(Cnode, '-------------------------' )
                
                # 2. CALCULATE TRUE TRIBUTARY LENGTH FOR THIS PILE
                if i == 0: 
                    z_below = sorted_pile_nodes[i+1][1][2]
                    trib_length = (z - z_below) / 2.0
                elif i == len(sorted_pile_nodes) - 1: 
                    z_above = sorted_pile_nodes[i-1][1][2]
                    trib_length = (z_above - z) / 2.0
                else: 
                    z_above = sorted_pile_nodes[i-1][1][2]
                    z_below = sorted_pile_nodes[i+1][1][2]
                    trib_length = (z_above - z_below) / 2.0
                    
                trib_length = max(trib_length, 0.001) 
                depth_from_surf = ground_surface_z - z 
                
                # 3. GET SOIL PROPERTIES
                current_layer, profile_z, used_bottom_layer_fallback = (
                    _soil_layer_for_model_elevation(z, active_soil_profile, profile_to_model_offset)
                )
                
                if not current_layer:
                    if int(Cnode) in bottom_node_tags:
                        message = (
                            f"No soil layer covers bottom node {int(Cnode)} at "
                            f"model z={z:.9g} (profile z={profile_z:.9g}); "
                            f"soil profile bounds are [{profile_bottom_z:.9g}, "
                            f"{profile_surface_z:.9g}]."
                        )
                        warnings.warn(message, RuntimeWarning, stacklevel=2)
                        raise ValueError(message)
                    soil_node_tag += 1
                    continue 

                if int(Cnode) in bottom_node_tags:
                    self.bottom_node_soil_assignments.append({
                        "node_tag": int(Cnode),
                        "model_z": float(z),
                        "depth_below_ground": float(depth_from_surf),
                        "profile_z": float(profile_z),
                        "layer_id": current_layer.get("layer_id"),
                        "uscs": current_layer.get("uscs"),
                        "soil_source": "bottom-layer fallback" if used_bottom_layer_fallback else "profile layer",
                    })
                    
                layer_type = current_layer['uscs']
                mat_type = 1 if layer_type[0].upper() in ['C','M'] else 2
                
                gamma = current_layer.get('gamma', 18000.0) # Fallback to 18 kN/m3
                sigV = max(depth_from_surf * gamma, 0.01)
                soil_logic = 'sand' if layer_type[0].upper() in ['S', 'G'] else 'clay'
                # if Cnode <3000:
                #     print(f" Linked Pile Node {Cnode} <---> Soil Node {Snode} | Coords: (X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}) | Soil Type: {layer_type} | Trib Length: {trib_length:.2f}")
                
                # ---------------------------------------------------------
                # 4. P-Y Springs (Lateral X & Y) with Parallel Radiation Damping
                # ---------------------------------------------------------
                pult, y50 = self._get_pile_soil_params('py', current_layer, depth_from_surf, pile_diameter, trib_length)
                self.store_spring_curve('py', pult/trib_length, y50, depth_from_surf, sigV, soil_logic)

                # A. Calculate Radiation Damping
                G_modulus = current_layer.get('G', 5000.0) # Fallback to 50e6 Pa if not found
                rho = gamma / 9.81  # Mass density
                Vs = math.sqrt(G_modulus / rho) # Shear wave velocity
                C_rad_per_m = 2.0 * pile_diameter * rho * Vs 
                
                # --- THE FIX: SCALE DOWN THE DAMPER ---
                # Theoretical damping is too aggressive against a fixed base.
                # Drop it to 2% just to kill the high-frequency ringing.
                damping_modifier = 0.0001
                C_rad = C_rad_per_m * trib_length * damping_modifier
                
                # B. Define Sequential Tags (Fits safely inside the +10 increment)
                spring_tag_x   = mat_tag_base + 1
                dashpot_tag_x  = mat_tag_base + 2   
                parallel_tag_x = mat_tag_base + 3  
                
                spring_tag_y   = mat_tag_base + 4
                dashpot_tag_y  = mat_tag_base + 5   
                parallel_tag_y = mat_tag_base + 6  
                
                tz_tag         = mat_tag_base + 7
                qz_tag         = mat_tag_base + 8

                # C. Build Parallel Materials (X and Y)
                op.uniaxialMaterial('PySimple1', spring_tag_x, mat_type, pult, y50, 0.3) 
                # op.uniaxialMaterial('Viscous', dashpot_tag_x, C_rad, 1.0) 
                # op.uniaxialMaterial('Parallel', parallel_tag_x, spring_tag_x, dashpot_tag_x)
                
                op.uniaxialMaterial('PySimple1', spring_tag_y, mat_type, pult, y50, 0.3) 
                # op.uniaxialMaterial('Viscous', dashpot_tag_y, C_rad, 1.0) 
                # op.uniaxialMaterial('Parallel', parallel_tag_y, spring_tag_y, dashpot_tag_y)
                
                # ---------------------------------------------------------
                # 5. T-Z Spring (Vertical Skin Friction)
                # ---------------------------------------------------------
                tult, z50 = self._get_pile_soil_params('tz', current_layer, depth_from_surf, pile_diameter, trib_length)
                self.store_spring_curve('tz', tult/trib_length, z50, depth_from_surf, sigV, soil_logic)

                op.uniaxialMaterial('TzSimple1', tz_tag, mat_type, tult, z50, 0.1)
                
                # Connect the fixed Snode to the moving Cnode (DOFs 1, 2, 3)
                # Assign parallel wrappers to lateral DOFs 1 & 2, and the standard T-Z spring to vertical DOF 3
                self.zeroEleTag += 1
                # op.element('zeroLength', int(self.zeroEleTag), Snode, Cnode, '-mat', parallel_tag_x, tz_tag, '-dir', 1,  3)
                op.element('zeroLength', int(self.zeroEleTag), Snode, Cnode, '-mat', spring_tag_x, spring_tag_y, tz_tag, '-dir', 1, 2, 3)
                self.zeroEleTags = np.append(self.zeroEleTags, self.zeroEleTag)
                if int(Cnode) in bottom_node_tags:
                    bottom_tz_nodes.add(int(Cnode))
                
                
                # ---------------------------------------------------------
                # 6. Q-Z Spring (Pile Tip Only)
                # ---------------------------------------------------------
                # Use the explicit bottom-node tags; socket bases and H-pile
                # tips do not share one elevation or one node-tag range.
                if int(Cnode) in bottom_node_tags:
                    qult, qz50 = self._get_pile_soil_params('qz', current_layer, depth_from_surf, pile_diameter, trib_length)
                    self.store_spring_curve('qz', qult, qz50, depth_from_surf, sigV, soil_logic)

                    op.uniaxialMaterial('QzSimple1', qz_tag, mat_type, qult, qz50, 0.1)
                    
                    # Apply strictly in the Z direction (DOF 3)
                    self.zeroEleTag += 1
                    op.element('zeroLength', int(self.zeroEleTag), Snode, Cnode, '-mat', qz_tag, '-dir', 3)
                    self.zeroEleTags = np.append(self.zeroEleTags, self.zeroEleTag)
                    bottom_qz_nodes.add(int(Cnode))
                    
                    
                    # Record this base node for dynamic excitation
                    self.BaseSoilNodes = np.append(self.BaseSoilNodes, Snode)
                    # 
                    print(f"    -> Added Q-Z End Bearing Spring at true Pile Tip (Z: {op.nodeCoord(Cnode)[2]:2f})")
                    print(f"    -> Added Q-Z End Bearing Spring at true Pile Tip Node ({Cnode}")
                    
                # Increment by 10 safely covers all 8 tags used above
                mat_tag_base += 10 
                soil_node_tag += 1

        missing_tz = bottom_node_tags.difference(bottom_tz_nodes)
        missing_qz = bottom_node_tags.difference(bottom_qz_nodes)
        if missing_tz or missing_qz:
            raise RuntimeError(
                "Bottom spring audit failed: "
                f"missing T-Z at {sorted(missing_tz)}; "
                f"missing Q-Z at {sorted(missing_qz)}"
            )
        print(
            f"Verified T-Z and Q-Z springs at all {len(bottom_node_tags)} "
            f"bottom nodes: {sorted(bottom_node_tags)}"
        )

        # Generates the Master Plots once all piles are finished
        # print(f'\n--- Generating Master Spring Plots  {self.BottomNodesSo} ---')
        self.plot_master_spring_curves()
       
       

   
    def _get_pile_soil_params(self, curve_type, layer_data, depth, b, pEleLength, gwtSwitch=1, loading_type='cyclic', pci_to_units=271447.1, psfin_to_units=1885.05, p_mult=0.5, seismic_softening_factor=0.15):
        """ 
        Calculates p-y, t-z, and q-z parameters strictly in N, m, s.
        Includes direct calibration hooks to match 3D target displacements.
        """
        import math
        import numpy as np

        # =====================================================================
        # CALIBRATION HOOKS - TUNE THESE TO MATCH YOUR 2.3 INCH TARGET
        # =====================================================================
        # To get more displacement, lower p_modifier and increase y_modifier.
        p_modifier = 0.40  # Drops ultimate soil strength to 30%
        y_modifier = 40.0   # Makes the soil 3x softer initially (increases y50)
        # =====================================================================

        uscs = layer_data.get('uscs', 'SM')
        gamma = layer_data.get('gamma', 0.01) # N/m^3
        phi = layer_data.get('phi', 0.0)      # degrees
        su = layer_data.get('su', 0.0)        # N/m^2 (Pa)
        eps50 = layer_data.get('eps50', 0.0)
        G = layer_data.get('G', 0.0)          # N/m^2 (Pa)
        
        k_py_user = layer_data.get('k_py', None)
        k_tz_user = layer_data.get('k_tz', None)

        pi = math.pi
        sigV = max(depth * gamma, 0.01)
        uscs_main = uscs[0].upper()
        soil_logic = 'sand' if uscs_main in ['S', 'G'] else 'clay'
        
        c_type = curve_type.strip().lower()

        if c_type == 'py':
            if soil_logic == 'sand':
                phi_rad = math.radians(phi)
                A = 0.9 
                K0 = 0.4
                Ka = math.tan(pi/4 - phi_rad/2)**2
                
                c1 = K0 * math.tan(phi_rad) * math.sin(pi/4 + phi_rad/2) / (math.tan(pi/4 - phi_rad/2) * math.cos(phi_rad/2))
                c2 = (math.tan(pi/4 + phi_rad/2)**2) * math.tan(phi_rad/2)
                c3 = K0 * math.tan(pi/4 + phi_rad/2) * (math.tan(phi_rad) * math.sin(pi/4 + phi_rad/2) - math.tan(phi_rad/2))
                c4 = (math.tan(pi/4 + phi_rad/2) / math.tan(pi/4 - phi_rad/2)) - Ka
                
                pst = gamma * depth * (depth * (c1 + c2 + c3) + b * c4)
                psd = b * gamma * depth * (Ka * (math.tan(pi/4 + phi_rad/2)**8 - 1) + K0 * math.tan(phi_rad) * math.tan(pi/4 + phi_rad/2)**4)
                
                pult_stress = A * min(pst, psd) if depth > 0 else 0.01
                pult_stress = pult_stress * p_mult
                
                if k_py_user is not None:
                    k_val = k_py_user * seismic_softening_factor
                else:
                    k_pci = 61.0 if gwtSwitch == 1 else 42.0
                    k_val = (k_pci * pci_to_units) * seismic_softening_factor
                
                y50 = 0.5 * (pult_stress / A) / (k_val * max(depth, 0.01)) * 0.5 * math.log((1+0.5)/(1-0.5))
                # print(f'y50 : {y50* y_modifier}, and p: {(pult_stress * pEleLength) * p_modifier}, at depth {depth} for sand')
                # --- APPLY CALIBRATION ---
                return (pult_stress * pEleLength) * p_modifier, y50 #* y_modifier
                
            else:
                pu_shallow = (3 + (sigV/su) + (0.5 * depth / b)) * su * b
                pu_deep = 9 * su * b
                p_stress = min(pu_shallow, pu_deep)
                p_stress = p_stress * p_mult
                
                if loading_type.lower() == 'cyclic' and depth < (2.5 * b):
                    p_stress *= 0.75 
                    
                y50 = 2.5 * eps50 * b
                # print(f'y50 : {y50* y_modifier}, and p: {(p_stress * pEleLength) * p_modifier}, at depth {depth} for sand')
                # --- APPLY CALIBRATION ---
                return (p_stress * pEleLength) * p_modifier, y50 #* y_modifier

        elif c_type == 'tz':
            if soil_logic == 'sand':
                delta = math.radians(0.8 * phi)
                tult = (0.4 * sigV * pi * b * math.tan(delta)) * pEleLength
                
                if k_tz_user is not None:
                    kf = k_tz_user
                else:
                    kf_psfin = np.interp(phi, [28, 31, 32, 34, 35, 38], [6000, 10000, 10000, 14000, 14000, 18000])
                    kf = kf_psfin * psfin_to_units
                
                z50 = tult / kf
                
                # --- APPLY CALIBRATION ---
                return tult * p_modifier, z50 #* y_modifier
                
            else:
                tult = (0.5 * su) * (pi * b) * pEleLength
                z50 = 0.01 * b  
                
                # --- APPLY CALIBRATION ---
                return tult * p_modifier, z50 #* y_modifier

        elif c_type == 'qz':
            if soil_logic == 'sand':
                phi_rad = math.radians(phi)
                Ir = G / (sigV * math.tan(phi_rad)) if G > 0 else 250
                Nq = (1+2*(1-math.sin(phi_rad)))*(1/(3-math.sin(phi_rad)))*math.exp(pi/2-phi_rad)*(math.tan(pi/4+phi_rad/2)**2)*(Ir**((4*math.sin(phi_rad))/(3*(1+math.sin(phi_rad)))))
                
                qult = (Nq * sigV) * (pi * b**2 / 4)
                qz50 = 0.125 * (0.05 * b)
                
                # --- APPLY CALIBRATION ---
                return qult * p_modifier, qz50 #* y_modifier
                
            else:
                qult = (9 * su) * (pi * b**2 / 4)
                qz50 = 0.05 * b
                
                # --- APPLY CALIBRATION ---
                return qult * p_modifier, qz50 #* y_modifier
                
        return 0.0, 0.0


    def store_spring_curve(self, curve_type, ult_force, disp50, depth, sigV, soil_logic):
        """ Calculates the backbone curve arrays and stores them in memory for later plotting. """
        import numpy as np

        if ult_force <= 0 or disp50 <= 0: 
            return

        ult_kN = ult_force / 1000.0
        disp50_mm = disp50 * 1000.0
        sigV_kPa = sigV / 1000.0

        disp_mm = np.linspace(0, 100, 100) # 0 to 100 mm displacement
        disp_m = disp_mm / 1000.0
        force_kN = np.zeros_like(disp_mm)
        c_type = curve_type.strip().lower()

        if c_type == 'py':
            if soil_logic == 'sand':
                force_kN = ult_kN * np.tanh(0.5493 * (disp_m / disp50))
            else:
                for i, y in enumerate(disp_m):
                    f = 0.5 * ult_kN * (y / disp50)**(1/3)
                    force_kN[i] = min(f, ult_kN)
        elif c_type in ['tz', 'qz']:
            for i, z in enumerate(disp_m):
                force_kN[i] = ult_kN * (z / (disp50 + z))

        self.spring_curves[c_type].append({
            'disp': disp_mm,
            'force': force_kN,
            'depth': depth,
            'sigV': sigV_kPa
        })


    def plot_master_spring_curves(self):
        """ Draws all stored curves onto 3 master plots (PY, TZ, QZ) and annotates them. """
        import os
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np

        plot_dir = "Spring_Plots"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        for c_type, curves in self.spring_curves.items():
            if not curves: 
                continue

            plt.figure(figsize=(10, 8))
            
            # Sort curves by depth so they plot nicely top-to-bottom
            curves = sorted(curves, key=lambda x: x['depth'])
            
            # Create a custom gradient from Pure Blue (shallow) to Pure Red (deep)
            from matplotlib.colors import LinearSegmentedColormap
            blue_to_red_cmap = LinearSegmentedColormap.from_list("blue_red", ["blue", "red"])
            colors = blue_to_red_cmap(np.linspace(0.0, 1.0, len(curves)))

            for i, curve in enumerate(curves):
                plt.plot(curve['disp'], curve['force'], color=colors[i], linewidth=1.5)
                
                annotation_text = f"{curve['depth']:.1f}m ({curve['sigV']:.0f}kPa)"
                plt.annotate(annotation_text, 
                             xy=(curve['disp'][-1], curve['force'][-1]), 
                             xytext=(5, 0),
                             textcoords="offset points", 
                             fontsize=8, 
                             color=colors[i],
                             verticalalignment='center')

            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel('Displacement (mm)', fontsize=16)
            plt.tick_params(axis='both', which='major', labelsize=14, length=8, width=1.5)
            plt.tick_params(axis='both', which='minor', labelsize=14, length=4, width=1)
            
            if c_type == 'py':
                plt.ylabel('Lateral Resistance, p (kN)', fontsize=16)
                plt.title('P-Y Curves', fontsize=16, fontweight='bold')
            elif c_type == 'tz':
                plt.ylabel('Vertical Skin Friction, t (kN)', fontsize=16)
                plt.title('T-Z Curves', fontsize=16, fontweight='bold')
            else:
                plt.ylabel('Tip Resistance, q (kN)', fontsize=16)
                plt.title('Q-Z Curves', fontsize=16, fontweight='bold')

            # plt.xlim(0, 115) 
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"Master_{c_type.upper()}_Curves.png"), dpi=200)
            plt.close()
            print(f" -> Saved Master {c_type.upper()} plot with {len(curves)} curves.")
    

    def getBaseSoilNodes(self):
        """Returns the list of fixed soil nodes at the pile tips."""
        return self.BaseSoilNodes.tolist()

