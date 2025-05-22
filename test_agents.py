#!/usr/bin/env python3
"""
Test script for the new Main Property Agent
This demonstrates how a single agent coordinates all tasks and returns simple JSON.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langcahin import process_client
import json

async def test_main_agent():
    """Test the main property agent with a sample client."""
    
    print("🏠 Testing Main Property Agent")
    print("=" * 50)
    
    try:
        # Test with client ID 691481
        client_id = 691481
        
        print(f"🎯 Processing Client ID: {client_id}")
        print("-" * 30)
        
        # Run the main agent (single call!)
        result_json = await process_client(client_id)
        result = json.loads(result_json)
        
        # Display simple results
        print("\n📊 FINAL RESULTS:")
        print("=" * 30)
        
        print(f"\n👤 CLIENT SUMMARY:")
        print(f"   {result.get('summary', 'No summary available')}")
        
        print(f"\n🏠 INVENTORY MATCH FOUND:")
        inventory_found = result.get('inventory_found', False)
        print(f"   {'✅ YES' if inventory_found else '❌ NO'}")
        
        print(f"\n📱 MESSAGE FOR CLIENT:")
        message = result.get('message', 'No message generated')
        print(f"   \"{message}\"")
        print(f"   Character Count: {len(message)}")
        print(f"   SMS Ready: {'✅ Yes' if len(message) <= 160 else '❌ Too long'}")
        
        # Check message quality
        has_placeholders = '[' in message or 'insert' in message.lower()
        print(f"   Human-like: {'✅ Yes' if not has_placeholders else '❌ Contains placeholders'}")
        
        print(f"\n🔧 Raw JSON Response:")
        print(json.dumps(result, indent=2))
        
        return result
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return None

def main():
    """Main function to run the test."""
    print("Starting Main Agent Test...")
    print("This agent uses other agents as tools internally!\n")
    
    # Run the async test
    results = asyncio.run(test_main_agent())
    
    if results:
        print("\n" + "="*50)
        print("✅ Test completed successfully!")
        print("🎉 Single agent coordinated everything and returned clean JSON!")
        
        # Show the simplicity
        summary = results.get('summary', '')
        message = results.get('message', '')
        found = results.get('inventory_found', False)
        
        print(f"\n📋 SUMMARY: Client profile analyzed")
        print(f"📨 MESSAGE: Ready to send to client")
        print(f"🏠 MATCH: {'Property found' if found else 'No match'}")
        
    else:
        print("\n❌ Test failed!")
    
    return results

if __name__ == "__main__":
    main() 