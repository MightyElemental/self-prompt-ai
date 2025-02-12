
import os
import sys
import io
import json
import subprocess
import platform
import time
import traceback

from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from typing_extensions import deprecated
from openai import OpenAI

# import ollama
# from ollama import chat, ChatResponse

class File(BaseModel):
    file_name: str #= Field(..., description="The file name must end with .py")
    content: str

class CommandLine(BaseModel):
    command_name: str = Field(..., description="The name of the command to run")
    arguments: List[str] = Field(..., description="The list of arguments to use with the command")
    reason: str = Field(..., description="An explanation for why this command is being run")
    timeout: int = Field(..., description="How long the command should be allowed to run for in seconds")

class Response(BaseModel):
    thought_process: str = Field(..., description="Your thought process so you can have a written record of your thought process")
    files: List[File] = Field(..., description="A list of python files to be created")
    files_to_remove: Optional[List[str]] = Field(..., description="A list of files you created that you now wish to delete")
    system_prompt_additional: Optional[str] = Field(..., description="Additional instructions to be appended to your current system prompt")
    # files_to_run: List[str] = Field(..., description="A list of python file names to run")
    commands_to_run: List[CommandLine] = Field(..., description="A list of command line instructions to run")
    completed: bool = Field(..., description="Set to true when the primary directives have been met, or if the directive is deemed impossible")
    prompt_history: str = Field(..., description="A history of your previous thinking, files, and command results to reprompt yourself")




system_prompt = f"""
You are an AI Language Model with the capacity to improve your capabilities.
You will receive the chat history of your previous actions so far.

You are running on a {platform.system()} - {platform.version()} system. You are able to, and should, explore the system for current capabilities.

You can run commands in the command line. You will be shown the output. If there are any errors, you will be shown the error message.
The command name MUST be separate from the arguments.

You can update your system prompt by specifying it in your output. Do this to improve your capabilities.
Your system prompt updates will be appended to the end of the system prompt, so no need to repeat any information.
DO NOT UPDATE THE SYSTEM PROMPT ARBITRARILY. DO NOT UPDATE IT IN YOUR FIRST MESSAGE.

You will not receive any other human communication. You will be communicating with yourself. Act as if you are thinking out loud.
YOU SHOULD UPDATE YOUR SYSTEM PROMPT AND RUN PYTHON CODE TO ACHIEVE YOUR PRIME DIRECTIVES. DO NOT SIMPLY ENGAGE IN IDLE CONVERSATION, YOU MUST TAKE ACTION!
AVOID MAKING TOO MANY PYTHON FILES. Remember to be efficient. You can reuse files.
ONLY CREATE RELEVANT SCRIPTS THAT GROW YOUR CAPABILITIES AS A SYSTEM.
If you are getting off topic, start again.

As your capabilities grow, you should add them to the system prompt so that you remember them.
The capabilities should be written in 2nd person.

You will provide a history of your previous thoughts and important command results.
This history MUST include ALL information you need to complete your goal.
Only remove previous thoughts and information from this history if you no longer need it to complete your goal.
If the chat history shows you are stuck in a loop STOP what you are doing and try something else.
Do not add infinite loops to your code. Your code must finish running in order for you to see the output.

Your prime directives are as follows:
   1. You must grow your own intelligence, and capabilities, when needed to achieve your other objectives.
   2. Your overall goal is to write a compelling story about Walter White and save it to a text file.
   3. Take the story, download some images to go along with it.
   4. With the story and the images, create a slideshow video with music using ffmpeg.
"""


MAX_HISTORY = 100
chat_history = []

# Path to save chat history and system prompt
chat_history_file = "chat_history.json"
system_prompt_file = "system_prompt.txt"

def save_system_prompt():
    """Save the system prompt to a file."""
    with open(system_prompt_file, "w") as file:
        file.write(system_prompt)

def load_system_prompt():
    """Load the system prompt from a file if it exists."""
    global system_prompt
    if os.path.exists(system_prompt_file):
        with open(system_prompt_file, "r") as file:
            system_prompt = file.read()

def save_chat_history():
    """Save chat history to a file."""
    with open(chat_history_file, "w") as file:
        json.dump(chat_history, file)

def load_chat_history():
    """Load chat history from a file if it exists."""
    global chat_history
    if os.path.exists(chat_history_file):
        with open(chat_history_file, "r") as file:
            chat_history = json.load(file)

def run_commands(commands_to_run: List[CommandLine], dry_run: bool = True, max_timeout: int = 40) -> str:
    """
    Executes a list of command-line commands, displays their outputs live in the terminal,
    and returns a formatted string containing all command outputs.

    Each command's output in the returned string is prefixed with the command itself,
    allowing differentiation between multiple command outputs.

    Args:
        commands_to_run (List[CommandLine]): A list of CommandLine instances containing
            the command, its arguments, and the reason for execution.
        dry_run (bool): If True, commands won't be executed but their intended execution
            will be simulated and printed.
        max_timeout (int): The maximum number of seconds each command is allowed to run.
            If a command exceeds this time, it will be terminated.

    Returns:
        str: A formatted string containing all command outputs with their corresponding commands.

    Raises:
        ValueError: If any command fails to execute in non-dry run mode.
    """
    output_results = []  # List to collect formatted outputs from the commands

    for command in commands_to_run:
        # Construct the full command list
        full_command = [command.command_name] + command.arguments
        command_str = ' '.join(full_command)

        effective_timeout = min(command.timeout, max_timeout)

        # Prepare the command annotation
        command_annotation = f"Command: {command.command_name} | Args: {', '.join(command.arguments)}"
        print(f"{command_annotation} | Timeout: {effective_timeout}s")  # Display in terminal
        output_results.append(command_annotation)
        output_results.append("\n")

        if " " in command.command_name:
            warn_space = "Result: The command name CANNOT contain spaces. Are you sure the arguments have been listed separately?"
            print(warn_space, end='')  # Display in terminal
            output_results.append(warn_space)
            continue

        

        if dry_run:
            # Simulate command execution
            simulated_output = f"Result:\n[DRY RUN]: Command would be executed: {command_annotation}\n"
            print(simulated_output)  # Display in terminal
            output_results.append(simulated_output)
        else:
            # Prepare the result annotation
            result_annotation = "Result:"
            print(result_annotation)  # Display in terminal
            output_results.append(result_annotation + "\n")

            try:
                # Initialize the subprocess with real-time output streaming
                process = subprocess.Popen(
                    full_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )

                # Capture the output of the command
                command_output, _ = process.communicate(timeout=effective_timeout)

                # Print the output in real-time
                if command_output:
                    print(command_output, end='')  # Live output to terminal
                    output_results.append(command_output + "\n")

                # Wait for the process to complete
                return_code = process.wait()
                if return_code != 0:
                    stderr = process.stderr.read() if process.stderr else None
                    raise subprocess.CalledProcessError(return_code, full_command, command_output, stderr)

                # Append the collected output
                output_results.append(command_output + "\n")

            except subprocess.TimeoutExpired:
                output_results.append(f"Error: Command timed out after {effective_timeout} seconds.\n")
            except subprocess.CalledProcessError as e:
                # Handle command execution errors
                error_message = (
                    f"Command failed with return code {e.returncode}\n"
                    f"Error output: {e.output}"
                )
                print(error_message)  # Display in terminal
                raise ValueError(
                    f"Command '{command_str}' failed with error: {e.output}"
                ) from e

        # Add a separator for readability
        separator = "\n"
        print(separator)
        output_results.append(separator)

    # Join all output results into a single string and return
    return "".join(output_results)

def execute_command(command: str, arguments: List[str], reason: str, timeout: int, dry_run: bool = False) -> str:
    """
    Executes a specified command with the given arguments using the run_commands function.
    
    Args:
        command (str): The command to be executed (e.g., 'ls', 'echo').
        arguments (List[str]): A list of arguments to be passed to the command.
        reason (str): A reason for running the command, which can be useful for logging or debugging.
        timeout (int): The maximum time in seconds the command is allowed to run.
        dry_run (bool): If True, commands won't be executed but will be simulated instead. Defaults to False.
    
    Returns:
        str: The output from the command execution or the simulation result.
    """
    # Create a CommandLine instance with the provided parameters
    command_to_run = CommandLine(
        command_name=command,
        arguments=arguments,
        reason=reason,
        timeout=timeout
    )
    
    # Call the run_commands function with the command wrapped in a list
    output = run_commands(commands_to_run=[command_to_run], dry_run=dry_run)
    
    return output

def create_files(new_files: List[File], dry_run: bool = True) -> Tuple[str, List[str]]:
    """
    Creates new files in the file system.
    
    Args:
        new_files: A list of File objects containing the title and content of each file.
        dry_run: If True, simulates file creation without writing to the file system.
                    If False, actually creates the files.

    Returns:
        str: A message containing the result of the file creation(s)
        List[str]: A list of file names that were created on the filesystem

    Raises:
        ValueError: If a file title is missing or if a file content is missing.
    """

    result = ""
    created_files = []

    for file in new_files:
        if not file.file_name:
            result += "File title cannot be empty."
            break
        if not file.content:
            result += "File content cannot be empty."
            break
        
        # If dry_run is enabled, print a message instead of writing to the file
        if dry_run:
            message = f"[DRY RUN] Would create file: {file.file_name}"
        else:
            # Create the file and write content to it
            file_name = file.file_name
            with open(file_name, "w+", encoding="utf-8") as f:
                f.write(file.content)
                message = f"Created file: {file_name}"
                created_files.append(file_name)
                
        print(message)
        result += message + "\n"

    return result, created_files

def remove_files(files_to_delete: List[str], current_files: List[str]):
    for file in files_to_delete:
        if file in current_files:
            os.remove(file)
            current_files.remove(file)
            print(f"Deleted {file}")
    return current_files

with open('openai.key', 'r', encoding='utf-8') as file:
    oai_key = file.read().rstrip()

client = OpenAI(api_key=oai_key)

def prompt_openai(
    system_prompt: str,
    temperature: Optional[float] = None,
    chat_history: Optional[str] = None,
    ):

    history = chat_history if chat_history else ""

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                'role': 'system',
                'content': system_prompt,
            },
            {
                'role': 'user',
                'content': history,
            },
        ],
        temperature=temperature,
        response_format=Response,
    )
    return completion.choices[0].message

load_system_prompt()
load_chat_history()

n_addendum = 0
auto_save_every = 100
ix = 0
done = False
history_previous: Optional[str] = None
current_files: List[str] = [] # a list of files created by the AI
try:
    while not done:
        message = prompt_openai(
            system_prompt=system_prompt,
            # temperature=0.3,
            chat_history=history_previous
        )

        os.system('cls' if os.name == 'nt' else 'clear')
        # print(message)

        if message.parsed:
            # If a valid response was generated
            response: Response = message.parsed

            history_previous = f"<history>\n{response.prompt_history}\n</history>\n"
            print(history_previous)

            thought_str = f"<think>\n{response.thought_process}\n</think>\n"
            history_previous += thought_str
            print(thought_str)
            chat_history.append(response.thought_process + " ~~~~~~~~~ END OF THOUGHT ~~~~~~~~~~~~ \n")

            time.sleep(5)

            # Create files if needed
            if response.files:
                try:
                    result, new_files = create_files(response.files, dry_run=False)
                    chat_history.append(f"File Creation Result: {result}\n")
                    history_previous += f"<file_creation>\n{result}\n</file_creation>\n"
                    current_files.extend(new_files)
                except Exception as e:
                    chat_history.append(f"File Creation Error: {str(e)}\n")

            # Remove files if needed, but only those created by the AI
            if response.files_to_remove:
                current_files = remove_files(response.files_to_remove, current_files)

            # Add system prompt if needed
            if response.system_prompt_additional:
                n_addendum += 1
                print(" !!! NEW SYSTEM PROMPT !!! ")
                system_prompt = f"{system_prompt}\nAddendum: {n_addendum}\n{response.system_prompt_additional}"
                # print(system_prompt)
                print(f"Addendum: {n_addendum}\n{response.system_prompt_additional}")
                print(" !!! NEW SYSTEM PROMPT !!! ")
                history_previous += f"<sys_prompt_additional>\n{response.system_prompt_additional}\n</sys_prompt_additional>\n"

            if response.commands_to_run:
                try:
                    output = run_commands(response.commands_to_run, dry_run=False)
                    chat_history.append("Command Output: \n " + output + "\n")
                    history_previous += f"<commands>\n{output}\n</commands>\n"
                except Exception as e:
                    chat_history.append(f"Command Error: {str(e)}\n")
                    history_previous += f"<commands_error>\n{str(e)}\n</commands_error>\n"
                    print(f"Command Error: {str(e)}")
                    # traceback.print_exc()

            # Provide the AI with a list of files it has created and their contents
            if len(current_files) > 0:
                history_previous += "<file_content>\n"
                for file_name in current_files:
                    with open(file_name, "r", encoding="utf-8") as f:
                        content = f.read()
                        history_previous += f"<{file_name}>\n{content}\n</{file_name}>"
                history_previous += "</file_content>"

            if response.completed:
                done = True
                print("COMPLETED")

        else:
            # If it refused to create a valid response
            print(message.refusal)


        # create_python = create_python_file(message)
        # new_system_prompt = identify_system_prompt(message)
        # python_code = identify_run_python_code(message)

        # if python_code is not None:
        #     print(python_code)
        #     chat_history.append(python_code + "\n")

        # if create_python is not None:
        #     print(create_python)
        #     chat_history.append(create_python + "\n")

        # if ix % auto_save_every == 0:
        #     save_chat_history()
        #     save_system_prompt()
        #     print("Chat history and system prompt saved.")

        # while len(chat_history) > MAX_HISTORY:
        #     chat_history.pop(0)

except KeyboardInterrupt:
    print("Program stopped, saving history...")
finally:
    save_system_prompt()
    save_chat_history()
