"""Tests for the ble_sdk.lead_handler module."""

import pytest
from unittest.mock import Mock, patch

from ble_sdk.handles.lead_handler import (
    parse_lead_status,
    create_lead_notification_handler,
    is_electrode_connected,
    get_disconnected_electrodes,
    FRAME_HEADER,
    FRAME_COMMAND
)


class TestLeadHandler:
    """Tests for the lead_handler module."""

    def setup_method(self):
        """Setup test data for each test method."""
        # Create a valid lead status data packet
        # Format: [Header(0xAA), Command(0xD8), Length(0x02), Cmd ID(0x0C), Status(0x03), 
        #          Lead data MSB, Lead data LSB, Checksum]
        self.valid_data = bytearray([FRAME_HEADER, FRAME_COMMAND, 0x02, 0x0C, 0x03, 0xA5, 0x5A, 0x00])
        
        # Lead status value: 0xA55A = 0b1010010101011010
        # P status (low 8 bits): 0b01011010 -> [0, 1, 0, 1, 1, 0, 1, 0]
        # N status (high 8 bits): 0b10100101 -> [1, 0, 1, 0, 0, 1, 0, 1]
        self.expected_p_status = [0, 1, 0, 1, 1, 0, 1, 0]
        self.expected_n_status = [1, 0, 1, 0, 0, 1, 0, 1]
        self.expected_raw_value = 0xA55A
        
        # Invalid data for testing
        self.invalid_length_data = bytearray([FRAME_HEADER, FRAME_COMMAND, 0x02, 0x0C])
        self.invalid_header_data = bytearray([0x00, 0x00, 0x02, 0x0C, 0x03, 0xA5, 0x5A, 0x00])

    def test_parse_lead_status_valid(self):
        """Test parsing valid lead status data."""
        result = parse_lead_status(self.valid_data)
        
        assert result["valid"] is True
        assert result["raw_value"] == self.expected_raw_value
        assert result["p_status"] == self.expected_p_status
        assert result["n_status"] == self.expected_n_status

    def test_parse_lead_status_invalid_length(self):
        """Test parsing data with invalid length."""
        result = parse_lead_status(self.invalid_length_data)
        
        assert result["valid"] is False
        assert result["raw_value"] == 0

    def test_parse_lead_status_invalid_header(self):
        """Test parsing data with invalid header."""
        result = parse_lead_status(self.invalid_header_data)
        
        assert result["valid"] is False
        assert result["raw_value"] == 0

    def test_is_electrode_connected(self):
        """Test checking if an electrode is connected."""
        lead_data = parse_lead_status(self.valid_data)
        
        # Check a few electrodes
        assert is_electrode_connected(lead_data, "1P") == self.expected_p_status[0]
        assert is_electrode_connected(lead_data, "2P") == self.expected_p_status[1]
        assert is_electrode_connected(lead_data, "1N") == self.expected_n_status[0]
        assert is_electrode_connected(lead_data, "2N") == self.expected_n_status[1]

    def test_is_electrode_connected_invalid_format(self):
        """Test checking with invalid electrode format."""
        lead_data = parse_lead_status(self.valid_data)
        
        with pytest.raises(ValueError):
            is_electrode_connected(lead_data, "X1")
            
        with pytest.raises(ValueError):
            is_electrode_connected(lead_data, "9P")
            
        with pytest.raises(ValueError):
            is_electrode_connected(lead_data, "AP")

    def test_get_disconnected_electrodes(self):
        """Test getting disconnected electrodes."""
        lead_data = parse_lead_status(self.valid_data)
        
        # Expected disconnected electrodes
        expected_disconnected = []
        for i in range(8):
            if self.expected_p_status[i] == 0:
                expected_disconnected.append(f"{i+1}P")
            if self.expected_n_status[i] == 0:
                expected_disconnected.append(f"{i+1}N")
        
        disconnected = get_disconnected_electrodes(lead_data)
        
        assert set(disconnected) == set(expected_disconnected)
        
    def test_get_disconnected_electrodes_invalid_data(self):
        """Test getting disconnected electrodes with invalid data."""
        lead_data = parse_lead_status(self.invalid_length_data)
        
        disconnected = get_disconnected_electrodes(lead_data)
        
        assert disconnected == []

    @patch('ble_sdk.lead_handler.logger')
    def test_create_lead_notification_handler(self, mock_logger):
        """Test creating a lead notification handler."""
        mock_callback = Mock()
        handler = create_lead_notification_handler(mock_callback)
        
        # Test with valid data
        handler(None, self.valid_data)
        
        mock_callback.assert_called_once()
        call_arg = mock_callback.call_args[0][0]
        assert call_arg["valid"] is True
        assert call_arg["raw_value"] == self.expected_raw_value
        
        # Test with invalid data
        mock_callback.reset_mock()
        handler(None, self.invalid_header_data)
        
        mock_callback.assert_not_called()
        mock_logger.warning.assert_called() 



  